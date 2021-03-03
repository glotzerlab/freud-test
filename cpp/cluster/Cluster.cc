// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <numeric>

#include "Cluster.h"
#include "NeighborBond.h"
#include "NeighborComputeFunctional.h"
#include "dset/dset.h"

//! Finds clusters using a network of neighbors.
namespace freud { namespace cluster {

void Cluster::compute(const freud::locality::NeighborQuery* nq, const freud::locality::NeighborList* nlist,
                      freud::locality::QueryArgs qargs, const unsigned int* keys)
{
    const unsigned int num_points = nq->getNPoints();
    m_cluster_idx.prepare(num_points);
    DisjointSets dj(num_points);

    freud::locality::loopOverNeighbors(
        nq, nq->getPoints(), num_points, qargs, nlist,
        [&dj](const freud::locality::NeighborBond& neighbor_bond) {
            // Merge the two sets using the disjoint set
            if (!dj.same(neighbor_bond.point_idx, neighbor_bond.query_point_idx))
            {
                dj.unite(neighbor_bond.point_idx, neighbor_bond.query_point_idx);
            }
        });

    // Done looping over points. All clusters are now determined.
    // Next, we renumber clusters from zero to num_clusters-1.
    // These new cluster indexes are then sorted by cluster size from largest
    // to smallest, with equally-sized clusters sorted based on their minimum
    // point index.
    std::vector<size_t> cluster_label(num_points, num_points);
    std::vector<size_t> cluster_label_count(num_points);
    std::vector<size_t> cluster_min_id(num_points, num_points);

    // Loop over every point.
    m_num_clusters = 0;
    for (size_t i = 0; i < num_points; i++)
    {
        size_t s = dj.find(i);

        // Label this cluster if we haven't seen it yet.
        if (cluster_label[s] == num_points)
        {
            // Label this cluster uniquely.
            cluster_label[s] = m_num_clusters;
            // Track the smallest point index in this cluster.
            cluster_min_id[cluster_label[s]] = i;
            // Increment the count of unique clusters.
            m_num_clusters++;
        }

        // Increment the counter for this cluster label.
        cluster_label_count[cluster_label[s]]++;
    }

    // Resize label counts and min ids to the number of unique clusters found.
    cluster_label_count.resize(m_num_clusters);
    cluster_label_count.shrink_to_fit();
    cluster_min_id.resize(m_num_clusters);
    cluster_min_id.shrink_to_fit();

    // Get a permutation that reorders clusters, largest to smallest.
    std::vector<size_t> cluster_reindex = sort_indexes_inverse(cluster_label_count, cluster_min_id);

    // Clear the cluster keys
    m_cluster_keys = std::vector<std::vector<unsigned int>>(m_num_clusters, std::vector<unsigned int>());

    /* Loop over all points, set their cluster ids and add them to a list of
     * sets. Each set contains all the keys that are part of that cluster. If
     * no keys are provided, the keys use point ids. Get the computed list
     * with getClusterKeys().
     */
    for (size_t i = 0; i < num_points; i++)
    {
        size_t s = dj.find(i);
        size_t cluster_idx = cluster_reindex[cluster_label[s]];
        m_cluster_idx[i] = cluster_idx;
        unsigned int key = i;
        if (keys != nullptr)
        {
            key = keys[i];
        }
        m_cluster_keys[cluster_idx].push_back(key);
    }
}

// Returns inverse permutation of cluster indexes, sorted from largest to smallest.
// Adapted from https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
std::vector<size_t> Cluster::sort_indexes_inverse(const std::vector<size_t>& counts,
                                                  const std::vector<size_t>& min_ids)
{
    // Initialize original index locations.
    std::vector<size_t> idx(counts.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Sort indexes based on comparing values in counts, min_ids.
    std::sort(idx.begin(), idx.end(), [&counts, &min_ids](size_t i1, size_t i2) {
        if (counts[i1] != counts[i2])
        {
            // If the counts are unequal, return the largest cluster first.
            return counts[i1] > counts[i2];
        }
        // If the counts are equal, return the cluster with the smallest
        // point id first.
        return min_ids[i1] < min_ids[i2];
    });

    // Invert the permutation.
    std::vector<size_t> inv_idx(idx.size());
    for (size_t i = 0; i < idx.size(); i++)
    {
        inv_idx[idx[i]] = i;
    }
    return inv_idx;
}

}; }; // end namespace freud::cluster
