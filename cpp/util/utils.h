#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <cmath>
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
template<class Scalar> inline Scalar modulusPositive(Scalar a, Scalar b)
{
    return std::fmod(std::fmod(a, b) + b, b);
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
