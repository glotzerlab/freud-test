// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "AngularSeparation.h"
#include "NeighborComputeFunctional.h"
#include "utils.h"

/*! \file AngularSeparation.cc
    \brief Compute the angular separation for each particle.
*/

namespace freud { namespace environment {

float computeSeparationAngle(const quat<float>& ref_q, const quat<float>& q)
{
    quat<float> R = q * conj(ref_q);
    auto theta = float(2.0 * std::acos(util::clamp(R.s, -1, 1)));
    return theta;
}

// The set of all equivalent quaternions equiv_qs is the set that takes the particle as it
// is defined to some global reference orientation. Thus, to be safe, we must include
// a rotation by qconst as defined below when doing the calculation.
// Important: equiv_qs must include both q and -q, for all included quaternions
float computeMinSeparationAngle(const quat<float>& ref_q, const quat<float>& q, const quat<float>* equiv_qs,
                                unsigned int n_equiv_quats)
{
    quat<float> qconst = equiv_qs[0];
    // here we undo a rotation represented by one of the equivalent orientations
    quat<float> qtemp = q * conj(qconst);

    // start with the quaternion before it has been rotated by equivalent rotations
    float min_angle = computeSeparationAngle(ref_q, q);

    // loop through all equivalent rotations and see if they have smaller angles with ref_q
    for (unsigned int i = 0; i < n_equiv_quats; ++i)
    {
        quat<float> qe = equiv_qs[i];
        quat<float> qtest = qtemp * qe;

        float angle_test = computeSeparationAngle(ref_q, qtest);

        if (angle_test < min_angle)
        {
            min_angle = angle_test;
        }
    }

    return min_angle;
}

void AngularSeparationNeighbor::compute(const locality::NeighborQuery* nq, const quat<float>* orientations,
                                        const vec3<float>* query_points,
                                        const quat<float>* query_orientations, unsigned int n_query_points,
                                        const quat<float>* equiv_orientations,
                                        unsigned int n_equiv_orientations,
                                        const freud::locality::NeighborList* nlist, locality::QueryArgs qargs)
{
    // This function requires a NeighborList object, so we always make one and store it locally.
    m_nlist = locality::makeDefaultNlist(nq, nlist, query_points, n_query_points, qargs);

    const size_t tot_num_neigh = m_nlist.getNumBonds();
    m_angles.prepare(tot_num_neigh);

    util::forLoopWrapper(0, nq->getNPoints(), [=](size_t begin, size_t end) {
        size_t bond(m_nlist.find_first_index(begin));
        for (size_t i = begin; i < end; ++i)
        {
            quat<float> q = orientations[i];

            for (; bond < tot_num_neigh && m_nlist.getNeighbors()(bond, 0) == i; ++bond)
            {
                const size_t j(m_nlist.getNeighbors()(bond, 1));
                quat<float> query_q = query_orientations[j];

                float theta = computeMinSeparationAngle(q, query_q, equiv_orientations, n_equiv_orientations);
                m_angles[bond] = theta;
            }
        }
    });
}

void AngularSeparationGlobal::compute(const quat<float>* global_orientations, unsigned int n_global,
                                      const quat<float>* orientations, unsigned int n_points,
                                      const quat<float>* equiv_orientations,
                                      unsigned int n_equiv_orientations)
{
    m_angles.prepare({n_points, n_global});

    util::forLoopWrapper(0, n_points, [=](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
        {
            quat<float> q = orientations[i];
            for (unsigned int j = 0; j < n_global; j++)
            {
                quat<float> global_q = global_orientations[j];
                float theta
                    = computeMinSeparationAngle(q, global_q, equiv_orientations, n_equiv_orientations);
                m_angles(i, j) = theta;
            }
        }
    });
}

}; }; // end namespace freud::environment
