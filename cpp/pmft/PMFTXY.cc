// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "PMFTXY.h"

/*! \file PMFTXY.cc
    \brief Routines for computing 2D potential of mean force in XY coordinates
*/

namespace freud { namespace pmft {

PMFTXY::PMFTXY(float x_max, float y_max, unsigned int n_x, unsigned int n_y) : PMFT()
{
    if (n_x < 1)
    {
        throw std::invalid_argument("PMFTXY requires at least 1 bin in X.");
    }
    if (n_y < 1)
    {
        throw std::invalid_argument("PMFTXY requires at least 1 bin in Y.");
    }
    if (x_max < 0)
    {
        throw std::invalid_argument("PMFTXY requires that x_max must be positive.");
    }
    if (y_max < 0)
    {
        throw std::invalid_argument("PMFTXY requires that y_max must be positive.");
    }

    // Note: There is an additional implicit volume factor of 2*pi
    // corresponding to the one rotational degree of freedom in the system.
    // However, this factor is implicitly canceled out since we also do not
    // include it in the number density computed for the system, see
    // PMFT::reduce for more information.
    const float dx = float(2.0) * x_max / float(n_x);
    const float dy = float(2.0) * y_max / float(n_y);
    m_jacobian = dx * dy;

    // Create the PCF array.
    m_pcf_array.prepare({n_x, n_y});

    // Construct the Histogram object that will be used to keep track of counts of bond distances found.
    BHAxes axes;
    axes.push_back(std::make_shared<util::RegularAxis>(n_x, -x_max, x_max));
    axes.push_back(std::make_shared<util::RegularAxis>(n_y, -y_max, y_max));
    m_histogram = BondHistogram(axes);
    m_local_histograms = BondHistogram::ThreadLocalHistogram(m_histogram);
}

void PMFTXY::reduce()
{
    float jacobian_factor = (float) 1.0 / m_jacobian;
    PMFT::reduce([jacobian_factor](size_t i) { return jacobian_factor; }); // NOLINT (misc-unused-parameters)
}

void PMFTXY::accumulate(const locality::NeighborQuery* neighbor_query, const float* query_orientations,
                        const vec3<float>* query_points, unsigned int n_query_points,
                        const locality::NeighborList* nlist, freud::locality::QueryArgs qargs)
{
    neighbor_query->getBox().enforce2D();
    accumulateGeneral(neighbor_query, query_points, n_query_points, nlist, qargs,
                      [=](const freud::locality::NeighborBond& neighbor_bond) {
                          vec3<float> delta(bondVector(neighbor_bond, neighbor_query, query_points));

                          // rotate interparticle vector
                          vec2<float> myVec(delta.x, delta.y);
                          rotmat2<float> myMat
                              = rotmat2<float>::fromAngle(-query_orientations[neighbor_bond.query_point_idx]);
                          vec2<float> rotVec = myMat * myVec;

                          m_local_histograms(rotVec.x, rotVec.y);
                      });
}

}; }; // end namespace freud::pmft
