// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <vector>

#include "LocalDescriptors.h"
#include "NeighborComputeFunctional.h"
#include "diagonalize.h"

/*! \file LocalDescriptors.cc
  \brief Computes local descriptors.
*/

namespace freud { namespace environment {

LocalDescriptors::LocalDescriptors(unsigned int l_max, bool negative_m,
                                   LocalDescriptorOrientation orientation)
    : m_l_max(l_max), m_negative_m(negative_m), m_nSphs(0), m_orientation(orientation)
{}

void LocalDescriptors::compute(const locality::NeighborQuery* nq, const vec3<float>* query_points,
                               unsigned int n_query_points, const quat<float>* orientations,
                               const freud::locality::NeighborList* nlist, locality::QueryArgs qargs,
                               unsigned int max_num_neighbors)
{
    // This function requires a NeighborList object, so we always make one and store it locally.
    m_nlist = locality::makeDefaultNlist(nq, nlist, query_points, n_query_points, qargs);

    if (max_num_neighbors == 0)
    {
        max_num_neighbors = std::numeric_limits<unsigned int>::max();
    }
    m_sphArray.prepare({m_nlist.getNumBonds(), getSphWidth()});

    util::forLoopWrapper(0, nq->getNPoints(), [=](size_t begin, size_t end) {
        fsph::PointSPHEvaluator<float> sph_eval(m_l_max);

        for (size_t i = begin; i < end; ++i)
        {
            size_t bond(m_nlist.find_first_index(i));
            unsigned int neighbor_count(0);

            vec3<float> rotation_0;
            vec3<float> rotation_1;
            vec3<float> rotation_2;

            if (m_orientation == LocalNeighborhood)
            {
                util::ManagedArray<float> inertiaTensor = util::ManagedArray<float>({3, 3});

                for (size_t bond_copy(bond); bond_copy < m_nlist.getNumBonds()
                     && m_nlist.getNeighbors()(bond_copy, 0) == i && neighbor_count < max_num_neighbors;
                     ++bond_copy, ++neighbor_count)
                {
                    const size_t j(m_nlist.getNeighbors()(bond_copy, 1));
                    const vec3<float> r_ij(bondVector(locality::NeighborBond(i, j), nq, query_points));
                    const float r_sq(dot(r_ij, r_ij));

                    for (size_t ii(0); ii < 3; ++ii)
                    {
                        inertiaTensor(ii, ii) += r_sq;
                    }

                    inertiaTensor(0, 0) -= r_ij.x * r_ij.x;
                    inertiaTensor(0, 1) -= r_ij.x * r_ij.y;
                    inertiaTensor(0, 2) -= r_ij.x * r_ij.z;
                    inertiaTensor(1, 0) -= r_ij.x * r_ij.y;
                    inertiaTensor(1, 1) -= r_ij.y * r_ij.y;
                    inertiaTensor(1, 2) -= r_ij.y * r_ij.z;
                    inertiaTensor(2, 0) -= r_ij.x * r_ij.z;
                    inertiaTensor(2, 1) -= r_ij.y * r_ij.z;
                    inertiaTensor(2, 2) -= r_ij.z * r_ij.z;
                }

                util::ManagedArray<float> eigenvalues = util::ManagedArray<float>(3);
                util::ManagedArray<float> eigenvectors = util::ManagedArray<float>({3, 3});

                freud::util::diagonalize33SymmetricMatrix(inertiaTensor, eigenvalues, eigenvectors);

                rotation_0 = vec3<float>(eigenvectors(0, 0), eigenvectors(0, 1), eigenvectors(0, 2));
                rotation_1 = vec3<float>(eigenvectors(1, 0), eigenvectors(1, 1), eigenvectors(1, 2));
                rotation_2 = vec3<float>(eigenvectors(2, 0), eigenvectors(2, 1), eigenvectors(2, 2));
            }
            else if (m_orientation == ParticleLocal)
            {
                const rotmat3<float> rotmat(conj(orientations[i]));
                rotation_0 = rotmat.row0;
                rotation_1 = rotmat.row1;
                rotation_2 = rotmat.row2;
            }
            else if (m_orientation == Global)
            {
                rotation_0 = vec3<float>(1, 0, 0);
                rotation_1 = vec3<float>(0, 1, 0);
                rotation_2 = vec3<float>(0, 0, 1);
            }
            else
            {
                throw std::runtime_error("Uncaught orientation mode in LocalDescriptors::compute");
            }

            neighbor_count = 0;
            for (; bond < m_nlist.getNumBonds() && m_nlist.getNeighbors()(bond, 0) == i
                 && neighbor_count < max_num_neighbors;
                 ++bond, ++neighbor_count)
            {
                const unsigned int sphCount(bond * getSphWidth());
                const size_t j(m_nlist.getNeighbors()(bond, 1));
                const vec3<float> r_ij(bondVector(locality::NeighborBond(i, j), nq, query_points));
                const float r_sq(dot(r_ij, r_ij));
                const vec3<float> bond_ij(dot(rotation_0, r_ij), dot(rotation_1, r_ij),
                                          dot(rotation_2, r_ij));

                const float magR(std::sqrt(r_sq));

                // Wrap theta into [0, 2*pi]
                float theta(std::atan2(bond_ij.y, bond_ij.x));
                theta = util::modulusPositive(theta, constants::TWO_PI);

                // Phi in [0, pi]
                float phi(std::acos(bond_ij.z / magR));

                // catch cases where bond_ij.z/magR falls outside [-1, 1]
                // due to numerical issues
                if (std::isnan(phi))
                {
                    phi = bond_ij.z > 0 ? 0 : M_PI;
                }

                sph_eval.compute(phi, theta);

                std::copy(sph_eval.begin(m_negative_m), sph_eval.end(), &m_sphArray[sphCount]);
            }
        }
    });

    // save the last computed number of particles
    m_nSphs = m_nlist.getNumBonds();
}

}; }; // end namespace freud::environment
