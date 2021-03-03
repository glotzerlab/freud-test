#ifndef THREADSTORAGE_H
#define THREADSTORAGE_H

#include "ManagedArray.h"
#include "utils.h"
#include <tbb/enumerable_thread_specific.h>
#include <vector>

namespace freud { namespace util {

//! Wrapper class for enumerable_thread_specific<T*>
/*! It is expected that default value for T is 0.
 */
template<typename T> class ThreadStorage
{
public:
    //! Default constructor
    ThreadStorage()
        : arrays(tbb::enumerable_thread_specific<ManagedArray<T>>([]() { return ManagedArray<T>(); }))
    {}

    //! Constructor with specific size for thread local arrays
    /*! \param size Size of the thread local arrays
     */
    explicit ThreadStorage(size_t size) : ThreadStorage(std::vector<size_t> {size}) {}

    //! Constructor with specific shape for thread local arrays
    /*! \param shape Vector of sizes in each dimension of the thread local arrays
     */
    explicit ThreadStorage(const std::vector<size_t>& shape)
        : arrays(
            tbb::enumerable_thread_specific<ManagedArray<T>>([shape]() { return ManagedArray<T>(shape); }))
    {}

    //! Destructor
    ~ThreadStorage() = default;

    //! Update size of the thread local arrays
    /*! \param size New size of the thread local arrays
     */
    void resize(size_t size)
    {
        resize(std::vector<size_t> {size});
    }

    //! Update size of the thread local arrays
    /*! \param size New size of the thread local arrays
     */
    void resize(std::vector<size_t> shape)
    {
        arrays
            = tbb::enumerable_thread_specific<ManagedArray<T>>([shape]() { return ManagedArray<T>(shape); });
    }

    //! Reset the contents of thread local arrays to be 0
    void reset()
    {
        for (auto array = arrays.begin(); array != arrays.end(); ++array)
        {
            array->reset();
        }
    }

    using const_iterator = typename tbb::enumerable_thread_specific<ManagedArray<T>>::const_iterator;
    using iterator = typename tbb::enumerable_thread_specific<ManagedArray<T>>::iterator;
    using reference = typename tbb::enumerable_thread_specific<ManagedArray<T>>::reference;

    const_iterator begin() const
    {
        return arrays.begin();
    }

    iterator begin()
    {
        return arrays.begin();
    }

    const_iterator end() const
    {
        return arrays.end();
    }

    iterator end()
    {
        return arrays.end();
    }

    reference local()
    {
        return arrays.local();
    }

    void reduceInto(ManagedArray<T>& result)
    {
        if (arrays.size() == 0)
        {
            // If no local arrays have been created, then no data can be reduced
            // and an error will occur if we attempt to iterate over arrays.
            // We simply reset the result array so it's all zeros.
            result.reset();
        }
        else
        {
            // Reduce over arrays into the result array.
            util::forLoopWrapper(0, result.size(), [=, &result](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i)
                {
                    for (auto arr = arrays.begin(); arr != arrays.end(); ++arr)
                    {
                        result[i] += (*arr)[i];
                    }
                }
            });
        }
    }

private:
    tbb::enumerable_thread_specific<ManagedArray<T>> arrays; //!< thread local arrays
};

}; }; // end namespace freud::util

#endif
