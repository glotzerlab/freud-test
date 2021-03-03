// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "PMFTR12.h"
#include "utils.h"

/*! \file PMFTR12.cc
    \brief Routines for computing potential of mean force and torque in R12 coordinates
*/

namespace freud { namespace pmft {

PMFTR12::PMFTR12(float r_max, unsigned int n_r, unsigned int n_t1, unsigned int n_t2) : PMFT()
{
    if (n_r < 1)
    {
        throw std::invalid_argument("PMFTR12 requires at least 1 bin in R.");
    }
    if (n_t1 < 1)
    {
        throw std::invalid_argument("PMFTR12 requires at least 1 bin in T1.");
    }
    if (n_t2 < 1)
    {
        throw std::invalid_argument("PMFTR12 requires at least 1 bin in T2.");
    }
    if (r_max < 0)
    {
        throw std::invalid_argument("PMFTR12 requires that r_max must be positive.");
    }

    // Construct the Histogram object that will be used to keep track of counts of bond distances found.
    BHAxes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(n_r, 0, r_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_t1, 0, constants::TWO_PI));
    axes.push_back(std::make_shared<util::RegularAxis>(n_t2, 0, constants::TWO_PI));
    m_histogram = BondHistogram(axes);
    m_local_histograms = BondHistogram::ThreadLocalHistogram(m_histogram);

    // Note: There is an additional implicit volume factor of 2*pi
    // corresponding to the rotational degree of freedom of the second particle
    // (i.e. both dt1 and dt2 technically have 2*pi in the numerator). This
    // factor is implicitly canceled out since we also do not include it in the
    // number density computed for the system. However, we do have to include
    // this factor for dt1 because it is part of the real space volume for the
    // central particle, see PMFT::reduce for more information.
    //
    // The array is computed as the inverse for faster use later.
    m_inv_jacobian_array.prepare({n_r, n_t1, n_t2});
    std::vector<float> bins_r = m_histogram.getBinCenters()[0];
    float dr = r_max / float(n_r);
    float dt1 = constants::TWO_PI / float(n_t1);
    float dt2 = 1 / float(n_t2);
    float product = dr * dt1 * dt2;
    for (unsigned int i = 0; i < n_r; i++)
    {
        float r = bins_r[i];
        for (unsigned int j = 0; j < n_t1; j++)
        {
            for (unsigned int k = 0; k < n_t2; k++)
            {
                m_inv_jacobian_array(i, j, k) = (float) 1.0 / (r * product);
            }
        }
    }

    // Create the PCF array.
    m_pcf_array.prepare({n_r, n_t1, n_t2});
}

void PMFTR12::reduce()
{
    PMFT::reduce([this](size_t i) { return m_inv_jacobian_array[i]; });
}

void PMFTR12::accumulate(const locality::NeighborQuery* neighbor_query, const float* orientations,
                         const vec3<float>* query_points, const float* query_orientations,
                         unsigned int n_query_points, const locality::NeighborList* nlist,
                         freud::locality::QueryArgs qargs)
{
    neighbor_query->getBox().enforce2D();
    accumulateGeneral(neighbor_query, query_points, n_query_points, nlist, qargs,
                      [=](const freud::locality::NeighborBond& neighbor_bond) {
                          vec3<float> delta(bondVector(neighbor_bond, neighbor_query, query_points));
                          // calculate angles
                          float d_theta1 = std::atan2(delta.y, delta.x);
                          float d_theta2 = std::atan2(-delta.y, -delta.x);
                          float t1 = orientations[neighbor_bond.point_idx] - d_theta1;
                          float t2 = query_orientations[neighbor_bond.query_point_idx] - d_theta2;
                          // make sure that t1, t2 are bounded between 0 and 2PI
                          t1 = util::modulusPositive(t1, constants::TWO_PI);
                          t2 = util::modulusPositive(t2, constants::TWO_PI);
                          m_local_histograms(neighbor_bond.distance, t1, t2);
                      });
}

}; }; // end namespace freud::pmft
