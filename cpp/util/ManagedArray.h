#ifndef MANAGED_ARRAY_H
#define MANAGED_ARRAY_H

#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

/*! \file ManagedArray.h
    \brief Defines the standard array class to be used throughout freud.
*/

namespace freud { namespace util {

//! Class to handle the storage of all arrays of numerical data used in freud.
/*! The purpose of this class is to handle standard memory management, and to
 *  provide an abstraction around the implementation-specific choice of
 *  underlying data structure for arrays of data in freud. These arrays are
 *  specifically designed for numerical data calculated by a compute class,
 *  particularly for arrays that must be made accessible through the Python API
 *  of freud. The array shape is stored and used to support multidimensional
 *  indexing.
 *
 *  To support resizing, a ManagedArray instances stores its data as a pointer
 *  to a pointer, and its shape as a pointer. As a result, copy-assignment or
 *  initialization will result in a new ManagedArray pointing to the same data,
 *  and any such array can resize or reallocate this data. The pointer to
 *  pointer infrastructure ensures that such changes properly propagate to all
 *  ManagedArrays referencing a given memory space. In addition, this
 *  infrastructure allows the creation of a completely new ManagedArray with a
 *  new set of pointers that also manages the same data, allowing it to keep
 *  the original array alive if the original ManagedArray instances become
 *  decoupled from it.
 *
 *  Performance notes:
 *      1. The variadic indexers may be a bottleneck if used in
 *         performance-critical code paths. In such cases, directly calling the
 *         overloaded signature using an std::vector is preferable.
 *      2. In situations where multiple identically shaped arrays are being
 *         indexed into, the index may be computed once using the getIndex
 *         function and reused to avoid recomputing it each time.
 */
template<typename T> class ManagedArray
{
public:
    //! Constructor based on a shape tuple.
    /*! Including a default value for the shape allows the usage of this
     *  constructor as the default constructor.
     *
     *  \param shape Shape of the array to allocate.
     */
    ManagedArray(const std::vector<size_t>& shape = {0})
    {
        prepare(shape, true);
    }

    //! Constructor based on a shape tuple.
    /*! Including a default value for the shape allows the usage of this
     *  constructor as the default constructor.
     *
     *  \param shape Shape of the array to allocate.
     */
    explicit ManagedArray(size_t size) : ManagedArray(std::vector<size_t> {size}) {}

    //! Destructor (currently empty because data is managed by shared pointer).
    ~ManagedArray() = default;

    //! Simple convenience for 1D arrays that calls through to the shape based `prepare` function.
    /*! \param new_size Size of the 1D array to allocate.
     */
    void prepare(size_t new_size)
    {
        prepare(std::vector<size_t> {new_size});
    }

    //! Prepare for writing new data.
    /*! This function always resets the array to contain zeros, but it will
     * also reallocate if there are other ManagedArrays pointing to the data in
     * order to ensure that those array references are not invalidated when
     * this function clears the data.
     *
     *  \param new_shape Shape of the array to allocate.
     *  \param force Reallocate regardless of whether anything changed or needs to be persisted.
     */
    void prepare(const std::vector<size_t>& new_shape, bool force = false)
    {
        // If we resized, or if there are outstanding references, we create a new array. No matter what,
        // reset.
        if (force || (m_data.use_count() > 1) || (new_shape != shape()))
        {
            m_shape = std::make_shared<std::vector<size_t>>(new_shape);

            m_size = std::make_shared<size_t>(1);
            for (unsigned int i = m_shape->size() - 1; i != static_cast<unsigned int>(-1); --i)
            {
                (*m_size) *= (*m_shape)[i];
            }

            // We make use of C-style arrays here rather than any alternative
            // because we need the underlying data representation to be
            // compatible with numpy on the Python side. We _could_ do this
            // with a different data structure like std::vector, but it would
            // require writing additional gymnastics to ensure proper reference
            // management and should be carefully considered before any rewrite.
            m_data = std::shared_ptr<std::shared_ptr<T>>(new std::shared_ptr<T>(
                new T[size()], std::default_delete<T[]>())); // NOLINT(modernize-avoid-c-arrays)
        }
        reset();
    }

    //! Reset the contents of array to be 0.
    void reset()
    {
        if (size() != 0)
        {
            memset((void*) get(), 0, sizeof(T) * size());
        }
    }

    //! Return a constant pointer to the underlying data (requires two levels of indirection).
    const T* get() const
    {
        std::shared_ptr<T>* tmp = m_data.get();
        return (*tmp).get();
    }

    //! Return the underlying pointer (requires two levels of indirection).
    /*! This function should only be used by client code when a raw pointer is
     * absolutely required. It is primarily part of the public API for the
     * purpose of freud's Python API, which requires a non-const pointer to the
     * data to construct a numpy array. There are specific use-cases (e.g.
     * interacting with Eigen) where directly accessing a mutable version of
     * the underlying data pointer is necessary, but users should be cautious
     * about overusing calls to get() rather than using the various operators
     * provided by the class.
     */
    T* get()
    {
        std::shared_ptr<T>* tmp = m_data.get();
        return (*tmp).get();
    }

    //! Writeable index into array.
    T& operator[](size_t index)
    {
        if (index >= size())
        {
            std::ostringstream msg;
            msg << "Attempted to access index " << index << " in an array of size " << size() << std::endl;
            throw std::invalid_argument(msg.str());
        }
        return get()[index];
    }

    //! Read-only index into array.
    const T& operator[](size_t index) const
    {
        if (index >= size())
        {
            std::ostringstream msg;
            msg << "Attempted to access index " << index << " in an array of size " << size() << std::endl;
            throw std::invalid_argument(msg.str());
        }
        return get()[index];
    }

    //! Get the size of the current array.
    size_t size() const
    {
        return *m_size;
    }

    //! Get the shape of the current array.
    std::vector<size_t> shape() const
    {
        return *m_shape;
    }

    //*************************************************************************
    // In order to support convenient indexing using arbitrary numbers of
    // indices, we provide overloads of the indexing operator.  All calls
    // eventually funnel through the simplest function interface, a std::vector
    // of indices. However, the more convenient approach is enabled using
    // variadic template arguments.
    //
    // Performance note: the variadic indexers can be expensive
    // since they have to build up a std::vector through a sequence of
    // (hopefully compiler optimized) function calls. In general, this is
    // unimportant since performance critical code paths in freud do not
    // involve large amounts of indexing into ManagedArrays. However, in cases
    // where critical code paths do index into the array, it may be
    // advantageous to directly call the function with a std::vector of indices
    // and bypass the variadic indexers.
    //*************************************************************************

    //! Implementation of variadic indexing function.
    template<typename... Ints> inline T& operator()(Ints... indices)
    {
        // cppcheck generates a false positive here on old machines (CI),
        // probably due to limited template support on those compilers.
        // cppcheck-suppress returnTempReference
        return (*this)(buildIndex(indices...));
    }

    //! Constant implementation of variadic indexing function.
    template<typename... Ints> inline const T& operator()(Ints... indices) const
    {
        // cppcheck generates a false positive here on old machines (CI),
        // probably due to limited template support on those compilers.
        // cppcheck-suppress returnTempReference
        return (*this)(buildIndex(indices...));
    }

    //! Core function for multidimensional indexing.
    /*! All the other convenience functions for indexing ultimately call this
     * function (or the const version below), which operates on a vector of
     * indexes.
     *
     * Note that the logic in getIndex is intentionally inlined here for
     * performance reasons. Although unimportant in most cases, operator() can
     * become a performance bottleneck when used in highly performance critical
     * code paths.
     */
    inline T& operator()(const std::vector<size_t>& indices)
    {
        size_t cur_prod = 1;
        size_t idx = 0;
        // In getting the linear bin, we must iterate over bins in reverse
        // order to build up the value of cur_prod because each subsequent axis
        // contributes less according to row-major ordering.
        for (unsigned int i = indices.size() - 1; i != static_cast<unsigned int>(-1); --i)
        {
            idx += indices[i] * cur_prod;
            cur_prod *= (*m_shape)[i];
        }
        return (*this)[idx];
    }

    //! Const version of core function for multidimensional indexing.
    inline const T& operator()(const std::vector<size_t>& indices) const
    {
        size_t cur_prod = 1;
        size_t idx = 0;
        // In getting the linear bin, we must iterate over bins in reverse
        // order to build up the value of cur_prod because each subsequent axis
        // contributes less according to row-major ordering.
        for (unsigned int i = indices.size() - 1; i != static_cast<unsigned int>(-1); --i)
        {
            idx += indices[i] * cur_prod;
            cur_prod *= (*m_shape)[i];
        }
        return (*this)[idx];
    }

    //! Get the multi-index corresponding to a single regular index.
    /*! This function is provided as an external utility in the event that
     * index generation is necessary without an actual array.
     *
     *  \param shape The shape to map indexes to.
     *  \param indices The index in each dimension.
     */
    static inline std::vector<size_t> getMultiIndex(const std::vector<size_t>& shape, size_t index)
    {
        size_t index_size = std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<>());

        std::vector<size_t> indices(shape.size());
        for (unsigned int i = 0; i < shape.size(); ++i)
        {
            index_size /= shape[i];
            // Integer division should cast away extras.
            indices[i] = index / index_size;
            index %= index_size;
        }
        return indices;
    }

    //! Get the linear index corresponding to a vector of indices in each dimension.
    /*! This function is provided as an external utility in the event that
     * index generation is necessary without an actual array.
     *
     *  \param shape The shape to map indexes to.
     *  \param indices The index in each dimension.
     */
    static inline size_t getIndex(const std::vector<size_t>& shape, const std::vector<size_t>& indices)
    {
        size_t cur_prod = 1;
        size_t idx = 0;
        // In getting the linear bin, we must iterate over bins in reverse
        // order to build up the value of cur_prod because each subsequent axis
        // contributes less according to row-major ordering.
        for (unsigned int i = indices.size() - 1; i != static_cast<unsigned int>(-1); --i)
        {
            idx += indices[i] * cur_prod;
            cur_prod *= shape[i];
        }
        return idx;
    }

    //! Get the linear index corresponding to a vector of indices in each dimension.
    /*! This function performs the conversion of a vector of indices
     *  into a linear index into the underlying data array of the ManagedArray.
     *  This function is primarily provided as a convenience, but may be useful
     *  to generate an index that can be reused multiple times.
     *
     *  \param indices The index in each dimension.
     */
    inline size_t getIndex(const std::vector<size_t>& indices) const
    {
        if (indices.size() != m_shape->size())
        {
            throw std::invalid_argument("Incorrect number of indices for this array.");
        }

        for (unsigned int i = 0; i < indices.size(); ++i)
        {
            if (indices[i] > (*m_shape)[i])
            {
                std::ostringstream msg;
                msg << "Attempted to access index " << indices[i] << " in dimension " << i
                    << ", which has size " << (*m_shape)[i] << std::endl;
                throw std::invalid_argument(msg.str());
            }
        }

        return getIndex(*m_shape, indices);
    }

    //! Return a copy of this array.
    /*! The returned object is a deep copy in the sense that it will copy every
     * element of the stored array. However, if the stored elements are
     * themselves pointers (e.g. if you create a ManagedArray<int*>), then the
     * copy will also retain pointers to that data.
     */
    ManagedArray copy() const
    {
        ManagedArray newarray(shape());
        for (unsigned int i = 0; i < size(); ++i)
        {
            newarray[i] = get()[i];
        }
        return newarray;
    }

private:
    //! The base case for building up the index.
    /*! These argument building functions are templated on two types, one that
     *  encapsulates the current object being operated on and the other being
     *  the list of remaining arguments. Since users may provide both signed and
     *  unsigned ints to the function, we perform the appropriate check on each
     *  Int object. The second function is used for template recursion in
     *  unwrapping the list of arguments.
     */
    template<typename Int> inline static std::vector<size_t> buildIndex(Int index)
    {
        return {static_cast<size_t>(index)};
    }

    //! The recursive case for building up the index (see above).
    template<typename Int, typename... Ints>
    inline static std::vector<size_t> buildIndex(Int index, Ints... indices)
    {
        std::vector<size_t> tmp = buildIndex(indices...);
        tmp.insert(tmp.begin(), static_cast<size_t>(index));
        return tmp;
    }

    std::shared_ptr<std::shared_ptr<T>> m_data;   //!< Pointer to array.
    std::shared_ptr<std::vector<size_t>> m_shape; //!< Shape of array.
    std::shared_ptr<size_t> m_size;               //!< Size of array.
};

}; }; // end namespace freud::util

#endif
