#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

namespace freud { namespace util {

//! Clip v if it is outside the range [lo, hi].
inline float clamp(float v, float lo, float hi)
{
    return std::max(lo, std::min(v, hi));
}

//! Modulus operation always resulting in a positive value
/*! \param a Dividend.
    \param b Divisor.
    \returns The remainder of a/b, between min(0, b) and max(0, b).
    \note This is the same behavior of the modulus operator % in Python (but not C++).
*/
template<typename Scalar> inline Scalar modulusPositive(Scalar a, Scalar b)
{
    return std::fmod(std::fmod(a, b) + b, b);
}

//! Sinc function, sin(x)/x
/*! \param x Argument.
    \returns Sin of x divided by x.
    \note There is no factor of pi in this definition (some conventions include pi).
*/
template<typename Scalar> inline Scalar sinc(Scalar x)
{
    if (x == 0)
    {
        return 1;
    }
    else
    {
        return std::sin(x) / x;
    }
}

//! Simpson's Rule numerical integration
/*! \param integrand Callable that returns the integrand value for a provided bin index.
    \param num_bins Number of bins to integrate over. Must be odd.
    \param dx Step size between bins.
    \note The integration summation is performed in double-precision regardless of Scalar type.
*/
template<typename Integrand> inline double simpson_integrate(Integrand& integrand, size_t num_bins, double dx)
{
    if (num_bins % 2 != 1)
    {
        // This only implements the easiest case of Simpson's rule.
        // Even numbers of bins require additional logic.
        throw std::invalid_argument("The number of integration bins must be odd.");
    }

    double integral = 0.0;

    // Simpson's rule uses prefactors 1, 4, 2, 4, 2, ..., 4, 1
    auto simpson_prefactor = [=](size_t bin) {
        if (bin == 0 || bin == num_bins - 1)
        {
            return 1.0;
        }
        else if (bin % 2 == 0)
        {
            return 2.0;
        }
        else
        {
            return 4.0;
        }
    };

    for (size_t bin_index = 0; bin_index < num_bins; bin_index++)
    {
        integral += simpson_prefactor(bin_index) * integrand(bin_index);
    }

    integral *= dx / 3.0;
    return integral;
}

//! Wrapper for for-loop to allow the execution in parallel or not.
/*! \param begin Beginning index.
 *  \param end Ending index.
 *  \param body An object with operator(size_t begin, size_t end).
 *  \param parallel If true, run body in parallel.
 */
template<typename Body>
inline void forLoopWrapper(size_t begin, size_t end, const Body& body, bool parallel = true)
{
    if (parallel)
    {
        tbb::parallel_for(tbb::blocked_range<size_t>(begin, end),
                          [&body](const tbb::blocked_range<size_t>& r) { body(r.begin(), r.end()); });
    }
    else
    {
        body(begin, end);
    }
}

//! Wrapper for 2D nested for loops to allow the execution in parallel or not.
/*! \param begin_row Beginning index of outer loop.
 *  \param end_row Ending index of outer loop.
 *  \param begin_col Beginning index of inner loop.
 *  \param end_col Ending index of inner loop.
 *  \param body An object with operator(size_t begin_row, size_t end_row, size_t begin_col, size_t end_col).
 *  \param parallel If true, run body in parallel.
 */
template<typename Body>
inline void forLoopWrapper2D(size_t begin_row, size_t end_row, size_t begin_col, size_t end_col,
                             const Body& body, bool parallel = true)
{
    if (parallel)
    {
        tbb::parallel_for(tbb::blocked_range2d<size_t>(begin_row, end_row, begin_col, end_col),
                          [&body](const tbb::blocked_range2d<size_t>& r) {
                              body(r.rows().begin(), r.rows().end(), r.cols().begin(), r.cols().end());
                          });
    }
    else
    {
        body(begin_row, end_row, begin_col, end_col);
    }
}

}; }; // namespace freud::util

#endif
