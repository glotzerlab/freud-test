// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef AABBQUERY_H
#define AABBQUERY_H

#include <cmath>
#include <map>
#include <memory>
#include <unordered_set>
#include <vector>

#include "AABBTree.h"
#include "Box.h"
#include "NeighborQuery.h"

/*! \file AABBQuery.h
 *  \brief Build an AABB tree from points and query it for neighbors.
 * A bounding volume hierarchy (BVH) tree is a binary search tree. It is
 * constructed from axis-aligned bounding boxes (AABBs). The AABB for a node in
 * the tree encloses all child AABBs. A leaf AABB holds multiple particles. The
 * tree is constructed in a balanced way using a heuristic to minimize AABB
 * volume. We build one tree per particle type, and use point AABBs for the
 * particles. The neighbor list is built by traversing down the tree with an
 * AABB that encloses the pairwise cutoff for the particle. Periodic boundaries
 * are treated by translating the query AABB by all possible image vectors,
 * many of which are trivially rejected for not intersecting the root node.
 */

namespace freud { namespace locality {

class AABBQuery : public NeighborQuery
{
public:
    //! Constructs the compute
    AABBQuery();

    //! New-style constructor.
    AABBQuery(const box::Box& box, const vec3<float>* points, unsigned int n_points);

    //! Destructor
    ~AABBQuery() override;

    //! Implementation of per-particle query for AABBQuery (see NeighborQuery.h for documentation).
    /*! \param query_point The point to find neighbors for.
     *  \param n_query_points The number of query points.
     *  \param qargs The query arguments that should be used to find neighbors.
     */
    std::shared_ptr<NeighborQueryPerPointIterator>
    querySingle(const vec3<float> query_point, unsigned int query_point_idx, QueryArgs args) const override;

    AABBTree m_aabb_tree; //!< AABB tree of points

protected:
    //! Validate the combination of specified arguments.
    /*! Add to parent function to account for the various arguments
     *  specifically required for AABBQuery nearest neighbor queries.
     */
    void validateQueryArgs(QueryArgs& args) const override
    {
        NeighborQuery::validateQueryArgs(args);
        if (args.mode == QueryType::nearest)
        {
            if (args.scale == DEFAULT_SCALE)
            {
                args.scale = float(1.1);
            }
            else if (args.scale <= float(1.0))
            {
                throw std::runtime_error("The scale query argument must be greater than 1.");
            }

            if (args.r_guess == DEFAULT_R_GUESS)
            {
                // By default, we assume a homogeneous system density and use
                // that to estimate the distance we need to query. This
                // calculation assumes a constant density of N/V, where N is
                // the number of particles and V is the box volume, and it
                // calculates the radius of a sphere that will contain the
                // desired number of neighbors.
                float r_guess = std::cbrtf(
                    (float(3.0) * static_cast<float>(args.num_neighbors) * m_box.getVolume())
                    / (float(4.0) * static_cast<float>(M_PI) * static_cast<float>(getNPoints())));

                // The upper bound is set by the minimum nearest plane distances.
                vec3<float> nearest_plane_distance = m_box.getNearestPlaneDistance();
                float min_plane_distance = std::min(nearest_plane_distance.x, nearest_plane_distance.y);
                if (!m_box.is2D())
                {
                    min_plane_distance = std::min(min_plane_distance, nearest_plane_distance.z);
                }

                args.r_guess = std::min(r_guess, min_plane_distance / float(2.0));
            }
            if (args.r_guess > args.r_max)
            {
                // No need to search past the requested bounds even if requested.
                args.r_guess = args.r_max;
            }
        }
    }

private:
    //! Driver for tree configuration
    void setupTree(unsigned int N);

    //! Maps particles by local id to their id within their type trees
    void mapParticlesByType();

    //! Driver to build AABB trees
    void buildTree(const vec3<float>* points, unsigned int N);

    std::vector<AABB> m_aabbs; //!< Flat array of AABBs of all types
};

//! Parent class of AABB iterators that knows how to traverse general AABB tree structures.
class AABBIterator : public NeighborQueryPerPointIterator
{
public:
    //! Constructor
    AABBIterator(const AABBQuery* neighbor_query, const vec3<float>& query_point,
                 unsigned int query_point_idx, float r_max, float r_min, bool exclude_ii)
        : NeighborQueryPerPointIterator(neighbor_query, query_point, query_point_idx, r_max, r_min,
                                        exclude_ii),
          m_aabb_query(neighbor_query)
    {}

    //! Empty Destructor
    ~AABBIterator() override = default;

    //! Computes the image vectors to query for
    void updateImageVectors(float r_max, bool _check_r_max = true);

protected:
    const AABBQuery* m_aabb_query;         //!< Link to the AABBQuery object
    std::vector<vec3<float>> m_image_list; //!< List of translation vectors
    unsigned int m_n_images {0};           //!< The number of image vectors to check
};

//! Iterator that gets a specified number of nearest neighbors from AABB tree structures.
class AABBQueryIterator : public AABBIterator
{
public:
    //! Constructor
    AABBQueryIterator(const AABBQuery* neighbor_query, const vec3<float>& query_point,
                      unsigned int query_point_idx, unsigned int num_neighbors, float r_guess, float r_max,
                      float r_min, float scale, bool exclude_ii)
        : AABBIterator(neighbor_query, query_point, query_point_idx, r_max, r_min, exclude_ii), m_count(0),
          m_num_neighbors(num_neighbors), m_search_extended(false), m_r_cur(r_guess), m_scale(scale),
          m_all_distances(), m_query_points_below_r_min()
    {
        updateImageVectors(0);
    }

    //! Empty Destructor
    ~AABBQueryIterator() override = default;

    //! Get the next element.
    NeighborBond next() override;

protected:
    unsigned int m_count;                          //!< Number of neighbors returned for the current point.
    unsigned int m_num_neighbors;                  //!< Number of nearest neighbors to find
    std::vector<NeighborBond> m_current_neighbors; //!< The current set of found neighbors.
    bool m_search_extended; //!< Flag to see whether we've gone past the safe cutoff distance and have to be
                            //!< worried about finding duplicates.
    float
        m_r_cur; //!< Current search ball cutoff distance in use for the current particle (expands as needed).
    float m_scale; //!< The amount to scale m_r by when the current ball is too small.
    std::map<unsigned int, float> m_all_distances; //!< Hash map of minimum distances found for a given point,
                                                   //!< used when searching beyond maximum safe AABB distance.
    std::unordered_set<unsigned int> m_query_points_below_r_min; //!< The set of query_points that were too
                                                                 //!< close based on the r_min threshold.
};

//! Iterator that gets neighbors in a ball of size r_max using AABB tree structures.
class AABBQueryBallIterator : public AABBIterator
{
public:
    //! Constructor
    AABBQueryBallIterator(const AABBQuery* neighbor_query, const vec3<float>& query_point,
                          unsigned int query_point_idx, float r_max, float r_min, bool exclude_ii,
                          bool _check_r_max = true)
        : AABBIterator(neighbor_query, query_point, query_point_idx, r_max, r_min, exclude_ii), cur_image(0),
          cur_node_idx(0), cur_ref_p(0)
    {
        updateImageVectors(m_r_max, _check_r_max);
    }

    //! Empty Destructor
    ~AABBQueryBallIterator() override = default;

    //! Get the next element.
    NeighborBond next() override;

private:
    unsigned int cur_image;    //!< The current node in the tree.
    unsigned int cur_node_idx; //!< The current node in the tree.
    unsigned int
        cur_ref_p; //!< The current index into the reference particles in the current node of the tree.
};
}; }; // end namespace freud::locality

#endif // AABBQUERY_H
