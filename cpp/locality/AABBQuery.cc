// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <stdexcept>

#include "AABBQuery.h"

namespace freud { namespace locality {

AABBQuery::AABBQuery(const box::Box& box, const vec3<float>* points, unsigned int n_points)
    : NeighborQuery(box, points, n_points)
{
    // Allocate memory and create image vectors
    setupTree(m_n_points);

    // Build the tree
    buildTree(m_points, m_n_points);
}

AABBQuery::~AABBQuery() = default;

std::shared_ptr<NeighborQueryPerPointIterator>
AABBQuery::querySingle(const vec3<float> query_point, unsigned int query_point_idx, QueryArgs args) const
{
    this->validateQueryArgs(args);
    if (args.mode == QueryType::ball)
    {
        return std::make_shared<AABBQueryBallIterator>(this, query_point, query_point_idx, args.r_max,
                                                       args.r_min, args.exclude_ii);
    }
    if (args.mode == QueryType::nearest)
    {
        return std::make_shared<AABBQueryIterator>(this, query_point, query_point_idx, args.num_neighbors,
                                                   args.r_guess, args.r_max, args.r_min, args.scale,
                                                   args.exclude_ii);
    }
    throw std::runtime_error("Invalid query mode provided to query function in AABBQuery.");
}

void AABBQuery::setupTree(unsigned int Np)
{
    m_aabbs.resize(Np);
}

void AABBQuery::buildTree(const vec3<float>* points, unsigned int Np)
{
    // Construct a point AABB for each point
    for (unsigned int i = 0; i < Np; ++i)
    {
        // Make a point AABB
        vec3<float> my_pos(points[i]);
        if (m_box.is2D())
        {
            my_pos.z = 0;
        }
        m_aabbs[i] = AABB(my_pos, i);
    }

    // Call the tree build routine, one tree per type
    m_aabb_tree.buildTree(m_aabbs.data(), Np);
}

void AABBIterator::updateImageVectors(float r_max, bool _check_r_max)
{
    box::Box box = m_neighbor_query->getBox();
    vec3<float> nearest_plane_distance = box.getNearestPlaneDistance();
    vec3<bool> periodic = box.getPeriodic();
    if (_check_r_max)
    {
        if ((periodic.x && nearest_plane_distance.x <= r_max * 2.0)
            || (periodic.y && nearest_plane_distance.y <= r_max * 2.0)
            || (!box.is2D() && periodic.z && nearest_plane_distance.z <= r_max * 2.0))
        {
            throw std::runtime_error("The AABBQuery r_max is too large for this box.");
        }
    }

    // Now compute the image vectors
    // Each dimension increases by one power of 3
    unsigned int n_dim_periodic = static_cast<unsigned int>(periodic.x)
        + static_cast<unsigned int>(periodic.y)
        + static_cast<unsigned int>(!box.is2D()) * static_cast<unsigned int>(periodic.z);
    m_n_images = 1;
    for (unsigned int dim = 0; dim < n_dim_periodic; ++dim)
    {
        m_n_images *= 3;
    }

    // Reallocate memory if necessary
    if (m_n_images > m_image_list.size())
    {
        m_image_list.resize(m_n_images);
    }

    vec3<float> latt_a = vec3<float>(box.getLatticeVector(0));
    vec3<float> latt_b = vec3<float>(box.getLatticeVector(1));
    vec3<float> latt_c = vec3<float>(0.0, 0.0, 0.0);
    if (!box.is2D())
    {
        latt_c = vec3<float>(box.getLatticeVector(2));
    }

    // There is always at least 1 image, which we put as our first thing to look at
    m_image_list[0] = vec3<float>(0.0, 0.0, 0.0);

    // Iterate over all other combinations of images
    unsigned int n_images = 1;
    for (int i = -1; i <= 1 && n_images < m_n_images; ++i)
    {
        for (int j = -1; j <= 1 && n_images < m_n_images; ++j)
        {
            for (int k = -1; k <= 1 && n_images < m_n_images; ++k)
            {
                if (!(i == 0 && j == 0 && k == 0))
                {
                    // Skip any periodic images if we don't have periodicity
                    if ((i != 0 && !periodic.x) || (j != 0 && !periodic.y)
                        || (k != 0 && (box.is2D() || !periodic.z)))
                    {
                        continue;
                    }

                    m_image_list[n_images] = float(i) * latt_a + float(j) * latt_b + float(k) * latt_c;
                    ++n_images;
                }
            }
        }
    }
}

NeighborBond AABBQueryBallIterator::next()
{
    float r_max_sq = m_r_max * m_r_max;
    float r_min_sq = m_r_min * m_r_min;

    // Read in the position of current point
    vec3<float> pos_i(m_query_point);
    if (m_neighbor_query->getBox().is2D())
    {
        pos_i.z = 0;
    }

    // Loop over image vectors
    while (cur_image < m_n_images)
    {
        // Make an AABB for the image of this point
        vec3<float> pos_i_image = pos_i + m_image_list[cur_image];
        AABBSphere asphere = AABBSphere(pos_i_image, m_r_max);

        // Stackless traversal of the tree
        while (cur_node_idx < m_aabb_query->m_aabb_tree.getNumNodes())
        {
            if (overlap(m_aabb_query->m_aabb_tree.getNodeAABB(cur_node_idx), asphere))
            {
                if (m_aabb_query->m_aabb_tree.isNodeLeaf(cur_node_idx))
                {
                    while (cur_ref_p < m_aabb_query->m_aabb_tree.getNodeNumParticles(cur_node_idx))
                    {
                        // Neighbor j
                        const unsigned int j
                            = m_aabb_query->m_aabb_tree.getNodeParticleTag(cur_node_idx, cur_ref_p);
                        // Increment before possible return.
                        cur_ref_p++;

                        // Skip ii matches immediately if requested.
                        if (m_exclude_ii && m_query_point_idx == j)
                        {
                            continue;
                        }

                        // Read in the position of j
                        vec3<float> pos_j((*m_neighbor_query)[j]);
                        if (m_neighbor_query->getBox().is2D())
                        {
                            pos_j.z = 0;
                        }

                        // Compute distance
                        const vec3<float> r_ij = pos_j - pos_i_image;
                        const float r_sq = dot(r_ij, r_ij);

                        // Check ii exclusion before including the pair.
                        if (r_sq < r_max_sq && r_sq >= r_min_sq)
                        {
                            return NeighborBond(m_query_point_idx, j, std::sqrt(r_sq));
                        }
                    }
                }
            }
            else
            {
                // Skip ahead
                cur_node_idx += m_aabb_query->m_aabb_tree.getNodeSkip(cur_node_idx);
            }
            cur_node_idx++;
            cur_ref_p = 0;
        } // end stackless search
        cur_image++;
        cur_node_idx = 0;
    } // end loop over images

    m_finished = true;
    return ITERATOR_TERMINATOR;
}

NeighborBond AABBQueryIterator::next()
{
    vec3<float> plane_distance = m_neighbor_query->getBox().getNearestPlaneDistance();
    float min_plane_distance = std::min(plane_distance.x, plane_distance.y);
    float max_plane_distance = std::max(plane_distance.x, plane_distance.y);
    if (!m_neighbor_query->getBox().is2D())
    {
        min_plane_distance = std::min(min_plane_distance, plane_distance.z);
        max_plane_distance = std::max(max_plane_distance, plane_distance.z);
    }

    // This iterator is not truly lazy; because it needs to get possible
    // neighbors and sort them to find the actual ones, it computes and caches
    // them and then returns them one-by-one. This check ensures that we only
    // search for new neighbors the first time next is called.
    if (m_current_neighbors.empty())
    {
        // Continually perform ball queries until the termination conditions are met.
        while (true)
        {
            // Perform a ball query to get neighbors. To ensure that we allow
            // ball queries to exceed their normal boundaries, we pass false as
            // the _check_r_max. We also can't depend on the ball query for
            // r_min filtering because we're querying beyond the normally safe
            // bounds, so we have to do it in this class.
            m_current_neighbors.clear();
            m_all_distances.clear();
            m_query_points_below_r_min.clear();
            std::shared_ptr<NeighborQueryPerPointIterator> ball_it = std::make_shared<AABBQueryBallIterator>(
                static_cast<const AABBQuery*>(m_neighbor_query), m_query_point, m_query_point_idx,
                std::min(m_r_cur, m_r_max), 0, m_exclude_ii, false);
            while (!ball_it->end())
            {
                NeighborBond nb = ball_it->next();
                if (nb == ITERATOR_TERMINATOR)
                {
                    continue;
                }

                if (!m_exclude_ii || m_query_point_idx != nb.point_idx)
                {
                    nb.query_point_idx = m_query_point_idx;
                    // If we've expanded our search radius beyond safe
                    // distance, use the map instead of the vector.
                    if (m_search_extended)
                    {
                        if ((m_all_distances.count(nb.point_idx) == 0)
                            || m_all_distances[nb.point_idx] > nb.distance)
                        {
                            m_all_distances[nb.point_idx] = nb.distance;
                            if (nb.distance < m_r_min)
                            {
                                m_query_points_below_r_min.insert(nb.point_idx);
                            }
                        }
                    }
                    else
                    {
                        if (nb.distance >= m_r_min)
                        {
                            m_current_neighbors.emplace_back(nb);
                        }
                    }
                }
            }

            // Break if there are enough neighbors, or if we are querying beyond the limits of
            // the periodic box.
            m_r_cur *= m_scale;

            if (m_current_neighbors.size() >= m_num_neighbors)
            {
                std::sort(m_current_neighbors.begin(), m_current_neighbors.end());
                break;
            }

            if ((m_r_cur >= m_r_max) || (m_r_cur >= max_plane_distance)
                || ((m_all_distances.size() - m_query_points_below_r_min.size()) >= m_num_neighbors))
            {
                // Once this condition is reached, either we found enough
                // neighbors beyond the normal min_plane_distance
                // condition or we conclude that there are not enough
                // neighbors left in the system.
                for (const auto& bond_distance : m_all_distances)
                {
                    if (bond_distance.second >= m_r_min)
                    {
                        m_current_neighbors.emplace_back(m_query_point_idx, bond_distance.first,
                                                         bond_distance.second);
                    }
                }
                std::sort(m_current_neighbors.begin(), m_current_neighbors.end());
                break;
            }

            if (m_r_cur > min_plane_distance / 2)
            {
                // If we have to go beyond the cutoff radius, we need to
                // start tracking what particles are already in the set so
                // that we can make sure that we find the closest image
                // because we now run the risk of finding duplicates.
                //
                // We could make this marginally more efficient by checking
                // whether we've exactly hit the limit, or if there's a
                // rescaling that would let us try the exact limit once
                // before going beyond the min plane distance.
                m_search_extended = true;
            }
        }
    }

    // Now we return all the points found for the current point, stopping when
    // any are beyond the maximum distance allowed.
    while ((m_count < m_num_neighbors) && (m_count < m_current_neighbors.size()))
    {
        m_count++;
        if (m_current_neighbors[m_count - 1].distance > m_r_max)
        {
            m_finished = true;
            return ITERATOR_TERMINATOR;
        }
        return m_current_neighbors[m_count - 1];
    }

    m_finished = true;
    return ITERATOR_TERMINATOR;
}

}; }; // end namespace freud::locality
