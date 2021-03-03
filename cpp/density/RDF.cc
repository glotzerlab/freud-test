// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "RDF.h"

/*! \file RDF.cc
    \brief Routines for computing radial density functions.
*/

namespace freud { namespace density {

RDF::RDF(unsigned int bins, float r_max, float r_min, bool normalize)
    : BondHistogramCompute(), m_normalize(normalize)
{
    if (bins == 0)
    {
        throw std::invalid_argument("RDF requires a nonzero number of bins.");
    }
    if (r_max <= 0)
    {
        throw std::invalid_argument("RDF requires r_max to be positive.");
    }
    if (r_max <= r_min)
    {
        throw std::invalid_argument("RDF requires that r_max must be greater than r_min.");
    }

    // Construct the Histogram object that will be used to keep track of counts of bond distances found.
    BHAxes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(bins, r_min, r_max));
    m_histogram = BondHistogram(axes);
    m_local_histograms = BondHistogram::ThreadLocalHistogram(m_histogram);

    // Precompute the cell volumes to speed up later calculations.
    m_vol_array2D.prepare(bins);
    m_vol_array3D.prepare(bins);
    float volume_prefactor = (float(4.0) / float(3.0)) * M_PI;
    std::vector<float> bin_boundaries = getBinEdges()[0];

    for (unsigned int i = 0; i < bins; i++)
    {
        float r = bin_boundaries[i];
        float nextr = bin_boundaries[i + 1];
        m_vol_array2D[i] = M_PI * (nextr * nextr - r * r);
        m_vol_array3D[i] = volume_prefactor * (nextr * nextr * nextr - r * r * r);
    }
}

void RDF::reduce()
{
    m_pcf.prepare(getAxisSizes()[0]);
    m_histogram.prepare(getAxisSizes()[0]);
    m_N_r.prepare(getAxisSizes()[0]);

    // Define prefactors with appropriate types to simplify and speed later code.
    float number_density = float(m_n_query_points) / m_box.getVolume();
    if (m_normalize)
    {
        number_density *= static_cast<float>(m_n_query_points - 1) / static_cast<float>(m_n_query_points);
    }
    auto np = static_cast<float>(m_n_points);
    auto nf = static_cast<float>(m_frame_counter);
    float prefactor = float(1.0) / (np * number_density * nf);

    util::ManagedArray<float> vol_array = m_box.is2D() ? m_vol_array2D : m_vol_array3D;
    m_histogram.reduceOverThreadsPerBin(m_local_histograms, [this, &prefactor, &vol_array](size_t i) {
        m_pcf[i] = m_histogram[i] * prefactor / vol_array[i];
    });

    // The accumulation of the cumulative density must be performed in
    // sequence, so it is done after the reduction.
    prefactor = float(1.0) / (np * static_cast<float>(m_frame_counter));
    m_N_r[0] = m_histogram[0] * prefactor;
    for (unsigned int i = 1; i < getAxisSizes()[0]; i++)
    {
        m_N_r[i] = m_N_r[i - 1] + m_histogram[i] * prefactor;
    }
}

void RDF::accumulate(const freud::locality::NeighborQuery* neighbor_query, const vec3<float>* query_points,
                     unsigned int n_query_points, const freud::locality::NeighborList* nlist,
                     freud::locality::QueryArgs qargs)
{
    accumulateGeneral(neighbor_query, query_points, n_query_points, nlist, qargs,
                      [=](const freud::locality::NeighborBond& neighbor_bond) {
                          m_local_histograms(neighbor_bond.distance);
                      });
}

}; }; // end namespace freud::density
