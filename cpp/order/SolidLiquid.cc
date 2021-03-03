// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "NeighborComputeFunctional.h"
#include "SolidLiquid.h"

namespace freud { namespace order {

SolidLiquid::SolidLiquid(unsigned int l, float q_threshold, unsigned int solid_threshold, bool normalize_q)
    : m_l(l), m_num_ms(2 * l + 1), m_q_threshold(q_threshold), m_solid_threshold(solid_threshold),
      m_normalize_q(normalize_q), m_steinhardt(l), m_cluster()
{
    if (m_q_threshold < 0.0)
    {
        throw std::invalid_argument(
            "SolidLiquid requires that the dot product cutoff q_threshold must be non-negative.");
    }
}

void SolidLiquid::compute(const freud::locality::NeighborList* nlist,
                          const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    // This function requires a NeighborList object, so we always make one and store it locally.
    m_nlist = locality::makeDefaultNlist(points, nlist, points->getPoints(), points->getNPoints(), qargs);

    const unsigned int num_query_points(m_nlist.getNumQueryPoints());

    // Compute Steinhardt using neighbor list (also gets ql for normalization)
    m_steinhardt.compute(&m_nlist, points, qargs);
    const auto& qlm = m_steinhardt.getQlm();
    const auto& ql = m_steinhardt.getQl();

    // Compute (normalized) dot products for each bond in the neighbor list
    const auto normalizationfactor = float(4.0 * M_PI / m_num_ms);
    const unsigned int num_bonds(m_nlist.getNumBonds());
    m_ql_ij.prepare(num_bonds);

    util::forLoopWrapper(
        0, num_query_points,
        [=](size_t begin, size_t end) {
            for (unsigned int i = begin; i != end; ++i)
            {
                unsigned int bond(m_nlist.find_first_index(i));
                for (; bond < num_bonds && m_nlist.getNeighbors()(bond, 0) == i; ++bond)
                {
                    const unsigned int j(m_nlist.getNeighbors()(bond, 1));

                    // Accumulate the dot product over m of qlmi and qlmj vectors
                    std::complex<float> bond_ql_ij = 0;
                    for (unsigned int k = 0; k < m_num_ms; k++)
                    {
                        bond_ql_ij += qlm(i, k) * std::conj(qlm(j, k));
                    }

                    // Optionally normalize dot products by points' ql values,
                    // accounting for the normalization of ql values
                    if (m_normalize_q)
                    {
                        bond_ql_ij *= normalizationfactor / (ql[i] * ql[j]);
                    }
                    m_ql_ij[bond] = bond_ql_ij.real();
                }
            }
        },
        true);

    // Filter neighbors to contain only solid-like bonds
    std::vector<bool> solid_filter(num_bonds);
    for (unsigned int bond(0); bond < num_bonds; bond++)
    {
        solid_filter[bond] = (m_ql_ij[bond] > m_q_threshold);
    }
    freud::locality::NeighborList solid_nlist(m_nlist);
    solid_nlist.filter(solid_filter.cbegin());

    // Save the neighbor counts of solid-like bonds for each query point
    m_number_of_connections.prepare(num_query_points);
    for (unsigned int i(0); i < num_query_points; i++)
    {
        m_number_of_connections[i] = solid_nlist.getCounts()[i];
    }

    // Filter nlist to only bonds between solid-like particles
    // (particles with more than solid_threshold solid-like bonds)
    const unsigned int num_solid_bonds(solid_nlist.getNumBonds());
    std::vector<bool> neighbor_count_filter(num_solid_bonds);
    for (unsigned int bond(0); bond < num_solid_bonds; bond++)
    {
        const unsigned int i(solid_nlist.getNeighbors()(bond, 0));
        const unsigned int j(solid_nlist.getNeighbors()(bond, 1));
        neighbor_count_filter[bond] = (m_number_of_connections[i] >= m_solid_threshold
                                       && m_number_of_connections[j] >= m_solid_threshold);
    }
    freud::locality::NeighborList solid_neighbor_nlist(solid_nlist);
    solid_neighbor_nlist.filter(neighbor_count_filter.cbegin());

    // Find clusters of solid-like particles
    m_cluster.compute(points, &solid_neighbor_nlist, qargs);
}

}; }; // end namespace freud::order
