#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <memory>
#include <utility>
#include <vector>
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#include <sstream>
#include <tbb/enumerable_thread_specific.h>
#include <utility>

#include "ManagedArray.h"
#include "utils.h"

namespace freud { namespace util {

//! Weight to add to a histogram.
/*! For histograms that are not simple counts, a Weight instance may be passed
 * in to indicate what value should be added to a bin. If not provided,
 * defaults to 1. Templated to allow floating weights if needed.
 */
template<typename T> struct Weight
{
    Weight() = default;
    explicit Weight(T value) : value(value), is_default(false) {}

    Weight& operator=(Weight other)
    {
        if (!is_default)
        {
            throw std::runtime_error("Weight can only be assigned once.");
        }
        value = other.value;
        is_default = false;
        return *this;
    }

    T value {1};
    bool is_default {true};
};

//! Class defining an axis of a histogram.
/*! An Axis is defined by a specified number of bins and the boundaries
 * defining them. Given a value along the Axis, the Axis can compute the bin
 * within which this value falls.
 */
class Axis
{
public:
    Axis() = default;

    virtual ~Axis() = default;

    Axis(size_t nbins, float min, float max) : m_nbins(nbins), m_min(min), m_max(max) {}

    // Return the number of bins in the histogram.
    size_t size() const
    {
        return m_nbins;
    }

    //! Find the bin of a value along this axis.
    /*! This method must be implemented by specific types of axes and is used
     * by the Histogram to bin along each axis independently.
     *
     * \param value The value to bin
     *
     * \return The index of the bin the value falls into.
     */
    virtual size_t bin(const float& value) const = 0;

    //! Return the boundaries of bins.
    std::vector<float> getBinEdges() const
    {
        return m_bin_edges;
    }

    //! Return the centers of bins.
    std::vector<float> getBinCenters() const
    {
        std::vector<float> bin_centers(m_nbins);
        for (size_t i = 0; i < m_nbins; i++)
        {
            bin_centers[i] = (m_bin_edges[i] + m_bin_edges[i + 1]) / float(2.0);
        }
        return bin_centers;
    }

    float getMin() const
    {
        return m_min;
    }

    float getMax() const
    {
        return m_max;
    }

    static const size_t OVERFLOW_BIN = 0xffffffff;

protected:
    size_t m_nbins {0};             //!< Number of bins
    float m_min;                    //!< Lowest value allowed.
    float m_max;                    //!< Highest value allowed.
    std::vector<float> m_bin_edges; //!< The edges of bins.
};

//! A regularly spaced axis.
/*! A RegularAxis is the most common type of axis, representing a series of
 * linearly spaced bins between two boundaries. These axes can be specified
 * with a small set of parameters and are very efficient when computing the
 * desired bin for a value. Given a value along the Axis, the Axis can compute
 * the bin within which this value falls.
 */
class RegularAxis : public Axis
{
public:
    RegularAxis(size_t nbins, float min, float max) : Axis(nbins, min, max)
    {
        m_bin_edges.resize(m_nbins + 1);
        m_bin_width = (max - min) / static_cast<float>(m_nbins);
        m_inverse_bin_width = float(1.0) / m_bin_width;
        // This must be <= because there is one more bin boundary than the number of bins.
        for (size_t i = 0; i <= nbins; i++)
        {
            // Spacing via multiplication is more numerically stable than
            // adding the bin width repeatedly
            m_bin_edges[i] = min + i * m_bin_width;
        }
    }

    ~RegularAxis() override = default;

    //! Find the bin of a value along this axis.
    /*! The linear spacing allows the binning process to be computed especially
     * efficiently since it simply reduces to scaling the value into the range
     * of the axis and looking at just the integral component of the resulting
     * value.
     *
     * \param value The value to bin
     *
     * \return The index of the bin the value falls into.
     */
    size_t bin(const float& value) const override
    {
        // Since we're using an unsigned int cast for truncation, we must
        // ensure that we will be working with a positive number or we will
        // fail to detect underflow.
        if (value < m_min || value >= m_max)
        {
            return OVERFLOW_BIN;
        }
        float val = (value - m_min) * m_inverse_bin_width;
        // fast float to int conversion with truncation
#ifdef __SSE2__
        size_t bin = _mm_cvtt_ss2si(_mm_load_ss(&val));
#else
        size_t bin = (size_t)(val);
#endif
        // Avoid rounding leading to overflow.
        if (bin == m_nbins)
        {
            return bin - 1;
        }
        return bin;
    }

protected:
    float m_bin_width;         //!< Bin width
    float m_inverse_bin_width; //!< Inverse of bin width
};

//! An n-dimensional histogram class.
/*! The Histogram is designed to simplify the most common use of histograms in
 * C++ code, which is looping over a series of values and then binning them. To
 * facilitate this use-case, the class provides an overriden operator() that
 * accepts a set of D values (for a histogram of dimensionality D), computes
 * the bin that value falls into, and then increments the count. Client code
 * can also directly request to know what bin a value would occupy if they wish
 * to operate on the histogram directly. The underlying data is handled using a
 * ManagedArray, allowing dispatch of the multi-dimensional indexing.
 */
template<typename T> class Histogram
{
public:
    //! A container for thread-local copies of a provided histogram.
    /*! This container implements the simplest method of enabling parallel-safe
     * accumulation, namely the creation of separate instances on each thread.
     * Thread local histograms can be accumulated later using the
     * reduceOverThreads functions in the Histogram class.
     *
     * To simplify the implementation and avoid unnecessary copies, the thread
     * local copies all share the same axes (because the axes are stored as
     * arrays of shared_ptrs in the Histogram class). This should cause no
     * problems, but can be refactored if needed.
     */
    class ThreadLocalHistogram
    {
    public:
        ThreadLocalHistogram() = default;

        explicit ThreadLocalHistogram(const Histogram& histogram)
            : m_local_histograms([histogram]() { return Histogram(histogram.m_axes); })
        {}

        using const_iterator = typename tbb::enumerable_thread_specific<Histogram<T>>::const_iterator;
        using iterator = typename tbb::enumerable_thread_specific<Histogram>::iterator;
        using reference = typename tbb::enumerable_thread_specific<Histogram>::reference;

        const_iterator begin() const
        {
            return m_local_histograms.begin();
        }

        iterator begin()
        {
            return m_local_histograms.begin();
        }

        const_iterator end() const
        {
            return m_local_histograms.end();
        }

        iterator end()
        {
            return m_local_histograms.end();
        }

        reference local()
        {
            return m_local_histograms.local();
        }

        void reset()
        {
            for (auto hist = m_local_histograms.begin(); hist != m_local_histograms.end(); ++hist)
            {
                hist->reset();
            }
        }

        //! Dispatch to thread local histogram.
        template<typename... FloatsOrWeight> void operator()(FloatsOrWeight... values)
        {
            m_local_histograms.local()(values...);
        }

        //! Dispatch to thread local histogram.
        void increment(size_t value_bin, T weight = 1)
        {
            m_local_histograms.local().increment(value_bin, weight);
        }

        // Reduce over histograms into the result array.
        void reduceInto(ManagedArray<T>& result)
        {
            result.reset();
            util::forLoopWrapper(0, result.size(), [=, &result](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i)
                {
                    for (auto hist = m_local_histograms.begin(); hist != m_local_histograms.end(); ++hist)
                    {
                        result[i] += hist->m_bin_counts[i];
                    }
                }
            });
        }

    protected:
        tbb::enumerable_thread_specific<Histogram<T>>
            m_local_histograms; //!< The thread-local copies of m_histogram.
    };

    using Axes = std::vector<std::shared_ptr<Axis>>;
    using AxisIterator = Axes::const_iterator;

    //! Default constructor
    Histogram() = default;

    //! Constructor
    explicit Histogram(std::vector<std::shared_ptr<Axis>> axes) : m_axes(std::move(axes))
    {
        std::vector<size_t> sizes(m_axes.size());
        std::transform(m_axes.begin(), m_axes.end(), sizes.begin(),
                       [](const auto& ax) { return ax->size(); });
        m_bin_counts = ManagedArray<T>(sizes);
    }

    //! Simple convenience for 1D arrays that calls through to the shape based `prepare` function.
    /*! \param new_size Size of the 1D array to allocate.
     */
    void prepare(size_t new_size)
    {
        prepare(std::vector<size_t> {new_size});
    }

    //! Prepare the underlying bin counts array.
    /*! Reallocate memory if needed.
     *
     *  \param new_shape Shape of the array to allocate.
     */
    void prepare(std::vector<size_t> new_shape)
    {
        m_bin_counts.prepare(new_shape);
    }

    //! Destructor
    ~Histogram() = default;

    //! Bin value and update the histogram count.
    template<typename... FloatsOrWeight> void operator()(FloatsOrWeight... values)
    {
        std::pair<std::vector<float>, Weight<T>> value_vector = getValueVector(values...);
        size_t value_bin = bin(value_vector.first);
        // Check for sentinel to avoid overflow.
        if (value_bin != Axis::OVERFLOW_BIN)
        {
            m_bin_counts[value_bin] += value_vector.second.value;
        }
    }

    //! Increment specified linear bin (with a specified weight if desired).
    void increment(size_t value_bin, T weight = 1)
    {
        // Check for sentinel to avoid overflow.
        if (value_bin != Axis::OVERFLOW_BIN)
        {
            m_bin_counts[value_bin] += weight;
        }
    }

    //! Find the bin of a value.
    /*! Bins are first computed along each axis of the histogram. These bins
     *  are then combined into a single linear index using the underlying
     *  ManagedArray.
     */
    size_t bin(std::vector<float> values) const
    {
        if (values.size() != m_axes.size())
        {
            std::ostringstream msg;
            msg << "This Histogram is " << m_axes.size() << "-dimensional, but " << values.size()
                << " values were provided in bin" << std::endl;
            throw std::invalid_argument(msg.str());
        }
        // First bin the values along each axis.
        std::vector<size_t> ax_bins;
        for (unsigned int ax_idx = 0; ax_idx < m_axes.size(); ++ax_idx)
        {
            size_t bin_i = m_axes[ax_idx]->bin(values[ax_idx]);
            // Immediately return sentinel if any bin is out of bounds.
            if (bin_i == Axis::OVERFLOW_BIN)
            {
                return Axis::OVERFLOW_BIN;
            }
            ax_bins.push_back(bin_i);
        }

        return m_bin_counts.getIndex(ax_bins);
    }

    //! Get the computed histogram.
    const ManagedArray<T>& getBinCounts() const
    {
        return m_bin_counts;
    }

    //! Get the shape of the computed histogram.
    std::vector<size_t> shape() const
    {
        return m_bin_counts.shape();
    }

    //! Reset the histogram.
    void reset()
    {
        m_bin_counts.reset();
    }

    //! Return the edges of bins.
    /*! This vector will be of size axis.size()+1 for each axis.
     */
    std::vector<std::vector<float>> getBinEdges() const
    {
        std::vector<std::vector<float>> bins(m_axes.size());
        for (unsigned int i = 0; i < m_axes.size(); ++i)
        {
            bins[i] = m_axes[i]->getBinEdges();
        }
        return bins;
    }

    //! Return the bin centers.
    /*! This vector will be of size axis.size() for each axis.
     */
    std::vector<std::vector<float>> getBinCenters() const
    {
        std::vector<std::vector<float>> bins(m_axes.size());
        for (unsigned int i = 0; i < m_axes.size(); ++i)
        {
            bins[i] = m_axes[i]->getBinCenters();
        }
        return bins;
    }

    //! Return a vector of tuples (min, max) indicating the bounds of each axis.
    std::vector<std::pair<float, float>> getBounds() const
    {
        std::vector<std::pair<float, float>> bounds(m_axes.size());
        for (unsigned int i = 0; i < m_axes.size(); ++i)
        {
            bounds[i] = std::pair<float, float>(m_axes[i]->getMin(), m_axes[i]->getMax());
        }
        return bounds;
    }

    //! Return a vector indicating the number of bins in each axis.
    std::vector<size_t> getAxisSizes() const
    {
        std::vector<size_t> sizes(m_axes.size());
        for (unsigned int i = 0; i < m_axes.size(); ++i)
        {
            sizes[i] = m_axes[i]->size();
        }
        return sizes;
    }

    //!< Aggregate a set of thread-local histograms into this one and apply a function.
    /*! This function can be used whenever reduction over a set of
     * ThreadLocalHistograms requires additional post-processing, such as some
     * sort of normalization per bin.
     *
     * \param local_histograms The set of local histograms to reduce into this one.
     * \param cf The function to apply to each bin, must have signature (size_t i) {...}
     */
    template<typename ComputeFunction>
    void reduceOverThreadsPerBin(ThreadLocalHistogram& local_histograms, const ComputeFunction& cf)
    {
        local_histograms.reduceInto(m_bin_counts);
        util::forLoopWrapper(0, m_bin_counts.size(), [=](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                cf(i);
            }
        });
    }

    //!< Aggregate a set of thread-local histograms into this one.
    /*! This function is the standard method for parallel aggregation of a
     * histogram. The simplest way to achieve parallel-safe accumulation is to
     * create a ThreadLocalHistogram object, which holds a local copy of a
     * histogram on each thread. This function then accumulates the results
     * into this object.
     *
     * \param local_histograms The set of local histograms to reduce into this one.
     */
    void reduceOverThreads(ThreadLocalHistogram& local_histograms)
    {
        // Simply call the per-bin function with a nullary function.
        reduceOverThreadsPerBin(local_histograms, [](size_t i) {});
    }

    //! Writeable index into array.
    T& operator[](size_t i)
    {
        return m_bin_counts[i];
    }

    //! Read-only index into array.
    const T& operator[](size_t i) const
    {
        return m_bin_counts[i];
    }

    size_t size() const
    {
        return m_bin_counts.size();
    }

protected:
    std::vector<std::shared_ptr<Axis>> m_axes; //!< The axes.
    ManagedArray<T> m_bin_counts;              //!< Counts for each bin

    //! The base case for type float when constructing a vector of values provided to operator().
    /*! This function and the accompanying recursive function below employ
     * variadic templating to accept an arbitrary set of float values and
     * construct a vector out of them.
     */
    std::pair<std::vector<float>, Weight<T>> getValueVector(float value) const
    {
        return {{value}, Weight<T>()};
    }

    //! The base case for type Weight when constructing a vector of values provided to operator().
    /*! This function and the accompanying recursive function below employ
     * variadic templating to accept an arbitrary set of float values and
     * construct a vector out of them.
     */
    std::pair<std::vector<float>, Weight<T>> getValueVector(Weight<T> weight) const
    {
        return {{}, weight};
    }

    //! The recursive case for constructing a vector of values (see base-case function docs).
    template<typename... FloatsOrWeight>
    std::pair<std::vector<float>, Weight<T>> getValueVector(float value, FloatsOrWeight... values) const
    {
        std::pair<std::vector<float>, Weight<T>> tmp = getValueVector(values...);
        tmp.first.insert(tmp.first.begin(), value);
        return tmp;
    }

    //! The recursive case for constructing a vector of values (see base-case function docs).
    template<typename... FloatsOrWeight>
    std::pair<std::vector<float>, Weight<T>> getValueVector(Weight<T> weight, FloatsOrWeight... values) const
    {
        std::pair<std::vector<float>, Weight<T>> tmp = getValueVector(values...);
        tmp.second = weight;
        return tmp;
    }
};

}; }; // namespace freud::util

#endif
