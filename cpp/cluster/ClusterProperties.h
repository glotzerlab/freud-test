// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CLUSTER_PROPERTIES_H
#define CLUSTER_PROPERTIES_H

#include "ManagedArray.h"
#include "NeighborQuery.h"

/*! \file ClusterProperties.h
    \brief Routines for computing properties of point clusters.
*/

namespace freud { namespace cluster {

//! Computes properties of clusters
/*! Given a set of points and \a cluster_idx (from Cluster, or some other
    source), ClusterProperties determines the following properties for each
    cluster:
     - Center of mass
     - Gyration tensor

    m_cluster_centers stores the computed center of mass for each cluster,
    properly handling periodic boundary conditions.
    m_cluster_gyrations stores a 3x3 gyration tensor for each cluster. The
    tensors are symmetric.
*/
class ClusterProperties
{
public:
    //! Constructor
    ClusterProperties() = default;

    //! Compute properties of the point clusters
    void compute(const freud::locality::NeighborQuery* nq, const unsigned int* cluster_idx);

    //! Get a reference to the last computed cluster centers
    const util::ManagedArray<vec3<float>>& getClusterCenters() const
    {
        return m_cluster_centers;
    }

    //! Get a reference to the last computed cluster gyration tensors
    const util::ManagedArray<float>& getClusterGyrations() const
    {
        return m_cluster_gyrations;
    }

    //! Get a reference to the last computed cluster size
    const util::ManagedArray<unsigned int>& getClusterSizes() const
    {
        return m_cluster_sizes;
    }

private:
    util::ManagedArray<vec3<float>>
        m_cluster_centers; //!< Center of mass computed for each cluster (length: m_num_clusters)
    util::ManagedArray<float>
        m_cluster_gyrations; //!< Gyration tensor computed for each cluster (m_num_clusters x 3 x 3 array)
    util::ManagedArray<unsigned int> m_cluster_sizes; //!< Size per cluster
};

}; }; // end namespace freud::cluster

#endif // CLUSTER_PROPERTIES_H
