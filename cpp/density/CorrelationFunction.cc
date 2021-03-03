// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <complex>
#include <stdexcept>
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "CorrelationFunction.h"
#include "NeighborBond.h"
#include "NeighborComputeFunctional.h"

/*! \file CorrelationFunction.cc
    \brief Generic pairwise correlation functions.
*/

namespace freud { namespace density {

template<typename T>
CorrelationFunction<T>::CorrelationFunction(unsigned int bins, float r_max) : BondHistogramCompute()
{
    if (bins == 0)
    {
        throw std::invalid_argument("CorrelationFunction  requires a nonzero number of bins.");
    }
    if (r_max <= 0)
    {
        throw std::invalid_argument("CorrelationFunction requires r_max to be positive.");
    }

    // We must construct two separate histograms, one for the counts and one
    // for the actual correlation function. The counts are used to normalize
    // the correlation function.
    util::Histogram<unsigned int>::Axes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(bins, 0, r_max));
    m_histogram = util::Histogram<unsigned int>(axes);
    m_local_histograms = util::Histogram<unsigned int>::ThreadLocalHistogram(m_histogram);

    typename util::Histogram<T>::Axes axes_rdf;
    axes_rdf.push_back(std::make_shared<util::RegularAxis>(bins, 0, r_max));
    m_correlation_function = util::Histogram<T>(axes_rdf);
    m_local_correlation_function = CFThreadHistogram(m_correlation_function);
}

//! \internal
//! helper function to reduce the thread specific arrays into one array
template<typename T> void CorrelationFunction<T>::reduce()
{
    m_histogram.prepare(getAxisSizes()[0]);
    m_correlation_function.prepare(getAxisSizes()[0]);

    // Reduce the bin counts over all threads, then use them to normalize the
    // RDF when computing.
    m_histogram.reduceOverThreads(m_local_histograms);
    m_correlation_function.reduceOverThreadsPerBin(m_local_correlation_function, [&](size_t i) {
        if (m_histogram[i])
        {
            m_correlation_function[i] /= m_histogram[i];
        }
    });
}

template<typename T> void CorrelationFunction<T>::reset()
{
    BondHistogramCompute::reset();

    // Zero the correlation function in addition to the bin counts that are
    // reset by the parent.
    m_local_correlation_function.reset();
}

// Define an overloaded pair of product functions to deal with complex conjugation if necessary.
inline std::complex<double> product(std::complex<double> x, std::complex<double> y)
{
    return std::conj(x) * y;
}

inline double product(double x, double y)
{
    return x * y;
}

template<typename T>
void CorrelationFunction<T>::accumulate(const freud::locality::NeighborQuery* neighbor_query, const T* values,
                                        const vec3<float>* query_points, const T* query_values,
                                        unsigned int n_query_points,
                                        const freud::locality::NeighborList* nlist,
                                        freud::locality::QueryArgs qargs)
{
    accumulateGeneral(
        neighbor_query, query_points, n_query_points, nlist, qargs,
        [=](const freud::locality::NeighborBond& neighbor_bond) {
            size_t value_bin = m_histogram.bin({neighbor_bond.distance});
            m_local_histograms.increment(value_bin);
            m_local_correlation_function.increment(
                value_bin,
                product(values[neighbor_bond.point_idx], query_values[neighbor_bond.query_point_idx]));
        });
}

template class CorrelationFunction<std::complex<double>>;
template class CorrelationFunction<double>;

}; }; // end namespace freud::density
