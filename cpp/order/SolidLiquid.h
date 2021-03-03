// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef SOLID_LIQUID_H
#define SOLID_LIQUID_H

#include <complex>
#include <iterator>
#include <vector>

#include "Cluster.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "Steinhardt.h"
#include "utils.h"

namespace freud { namespace order {

//! Identifies solid-like clusters using dot products of q_{lm}.
/*! The solid-liquid order parameter (ten Wolde 1995) uses a Steinhardt-like
 *  approach to identify solid-like particles. First, a bond parameter
 *  q_l(i, j) is computed for each neighbor bond.
 *
 *  If normalize_q is true (default), the bond parameter is given by
 *  q_l(i, j) = \frac{\sum_{m=-l}^{l} \text{Re}~q_{lm}(i) q_{lm}^*(j)}
 *  {\sqrt{\sum_{m=-l}^{l} \lvert q_{lm}(i) \rvert^2}
 *  \sqrt{\sum_{m=-l}^{l} \lvert q_{lm}(j) \rvert^2}}
 *
 *  If normalize_q is false, then the denominator of the above
 *  expression is left out.
 *
 *  Next, the bonds are filtered to keep only "solid-like" bonds with
 *  q_l(i, j) above a cutoff value q_{threshold}.
 *
 *  If a particle has more than solid_threshold solid-like bonds, then
 *  the particle is considered solid-like. Finally, solid-like particles are
 *  clustered.
 *
 *  References:
 *  ten Wolde, P. R., Ruiz-Montero, M. J., & Frenkel, D. (1995).
 *  Numerical Evidence for bcc Ordering at the Surface of a Critical fcc Nucleus.
 *  Phys. Rev. Lett., 75 (2714). https://doi.org/10.1103/PhysRevLett.75.2714
 *
 *  Filion, L., Hermes, M., Ni, R., & Dijkstra, M. (2010).
 *  Crystal nucleation of hard spheres using molecular dynamics, umbrella sampling,
 *  and forward flux sampling: A comparison of simulation techniques.
 *  J. Chem. Phys. 133 (244115). https://doi.org/10.1063/1.3506838
 */

class SolidLiquid
{
public:
    //! Constructor
    /*! Constructor for Solid-Liquid analysis class.
     *  \param l Spherical harmonic number l.
     *  \param q_threshold Value of dot product threshold when evaluating
     *     \f$Q_{lm}(i) Q_{lm}^*(j)\f$ to determine if a neighbor pair is
     *     a solid-like bond. (For l=6, 0.7 is generally good for FCC or BCC
     *     structures)
     *  \param solid_threshold Minimum required number of adjacent solid-like bonds
     *     for a particle to be considered solid-like for clustering. (For
     *     l=6, 6-8 is generally good for FCC or BCC structures)
     *  \param normalize_q Whether to normalize the per-bond dot products of qlm.
     */
    SolidLiquid(unsigned int l, float q_threshold, unsigned int solid_threshold, bool normalize_q = true);

    unsigned int getL() const
    {
        return m_l;
    }

    float getQThreshold() const
    {
        return m_q_threshold;
    }

    unsigned int getSolidThreshold() const
    {
        return m_solid_threshold;
    }

    bool getNormalizeQ() const
    {
        return m_normalize_q;
    }

    //! Compute the Solid-Liquid Order Parameter
    void compute(const freud::locality::NeighborList* nlist, const freud::locality::NeighborQuery* points,
                 freud::locality::QueryArgs qargs);

    //! Returns largest cluster size.
    unsigned int getLargestClusterSize() const
    {
        return m_cluster.getClusterKeys()[0].size();
    }

    //! Returns a vector containing the size of all clusters.
    std::vector<unsigned int> getClusterSizes() const
    {
        auto keys = m_cluster.getClusterKeys();
        std::vector<unsigned int> sizes;
        sizes.reserve(keys.size());
        std::transform(keys.begin(), keys.end(), std::back_inserter(sizes),
                       [](auto& key) { return key.size(); });
        return sizes;
    }

    //! Get a reference to the last computed set of solid-like cluster
    //  indices for each particle
    const util::ManagedArray<unsigned int>& getClusterIdx() const
    {
        return m_cluster.getClusterIdx();
    }

    //! Get a reference to the number of connections per particle
    const util::ManagedArray<unsigned int>& getNumberOfConnections() const
    {
        return m_number_of_connections;
    }

    unsigned int getNumClusters() const
    {
        return m_cluster.getNumClusters();
    }

    //! Return a pointer to the NeighborList used in the last call to compute.
    locality::NeighborList* getNList()
    {
        return &m_nlist;
    }

    //! Return the ql_ij values.
    const util::ManagedArray<float>& getQlij() const
    {
        return m_ql_ij;
    }

private:
    unsigned int m_l;               //!< Value of l for the spherical harmonic.
    unsigned int m_num_ms;          //!< The number of magnetic quantum numbers (2*m_l+1).
    float m_q_threshold;            //!< Dot product cutoff
    unsigned int m_solid_threshold; //!< Solid-like num connections cutoff
    bool m_normalize_q;             //!< Whether to normalize the qlmi dot products.
    locality::NeighborList m_nlist; //!< The NeighborList used in the last call to compute.

    freud::order::Steinhardt m_steinhardt; //!< Steinhardt class used to compute qlm
    freud::cluster::Cluster m_cluster;     //!< Cluster class used to cluster solid-like bonds

    util::ManagedArray<float> m_ql_ij;                        //!< All of the qlmi dot qlmj's computed
    util::ManagedArray<unsigned int> m_number_of_connections; //! Number of connections for each particle with
                                                              //! dot product above q_threshold
};

}; }; // end namespace freud::order

#endif // SOLID_LIQUID_H
