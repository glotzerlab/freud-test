// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iterator>
#include <tbb/parallel_sort.h>
#include <vector>

#include "NeighborBond.h"
#include "Voronoi.h"

/*! \file Voronoi.cc
    \brief Computes Voronoi neighbors for a set of points.
*/

namespace freud { namespace locality {

// Voronoi calculations should be kept in double precision.
void Voronoi::compute(const freud::locality::NeighborQuery* nq)
{
    const auto box = nq->getBox();
    const auto n_points = nq->getNPoints();

    m_polytopes.resize(n_points);
    m_volumes.prepare(n_points);

    const vec3<float> v1 = box.getLatticeVector(0);
    const vec3<float> v2 = box.getLatticeVector(1);
    const vec3<float> v3 = (box.is2D() ? vec3<float>(0, 0, 1) : box.getLatticeVector(2));

    // This heuristic for choosing blocks is based on the voro::pre_container
    // guess_optimal method. By computing the heuristic directly, we avoid
    // having to create a pre_container. This saves time because the
    // pre_container cannot be used to set up container_periodic (only
    // non-periodic containers are compatible).
    const float block_scale
        = std::pow(n_points / (voro::optimal_particles * box.getVolume()), float(1.0 / 3.0));
    const int voro_blocks_x = int(box.getLx() * block_scale + 1);
    const int voro_blocks_y = int(box.getLy() * block_scale + 1);
    const int voro_blocks_z = int(box.getLz() * block_scale + 1);

    voro::container_periodic container(v1.x, v2.x, v2.y, v3.x, v3.y, v3.z, voro_blocks_x, voro_blocks_y,
                                       voro_blocks_z, 3);

    for (size_t query_point_id = 0; query_point_id < n_points; query_point_id++)
    {
        vec3<double> query_point((*nq)[query_point_id]);
        container.put(query_point_id, query_point.x, query_point.y, query_point.z);
    }

    voro::voronoicell_neighbor cell;
    voro::c_loop_all_periodic voronoi_loop(container);
    std::vector<double> face_areas;
    std::vector<int> face_vertices;
    std::vector<int> neighbors;
    std::vector<double> normals;
    std::vector<double> vertices;
    std::vector<NeighborBond> bonds;

    if (voronoi_loop.start())
    {
        do
        {
            container.compute_cell(cell, voronoi_loop);

            // Get id and position of current particle
            const int query_point_id(voronoi_loop.pid());
            vec3<double> query_point(voronoi_loop.x(), voronoi_loop.y(), voronoi_loop.z());

            // Get Voronoi cell properties
            cell.face_areas(face_areas);
            cell.face_vertices(face_vertices);
            cell.neighbors(neighbors);
            cell.normals(normals);
            cell.vertices(query_point.x, query_point.y, query_point.z, vertices);

            // Compute polytope vertices in relative coordinates
            std::vector<vec3<double>> relative_vertices;
            auto vertex_iterator = vertices.begin();
            while (vertex_iterator != vertices.end())
            {
                double vert_x = *vertex_iterator;
                vertex_iterator++;
                double vert_y = *vertex_iterator;
                vertex_iterator++;
                double vert_z = *vertex_iterator;
                vertex_iterator++;

                // In 2D systems, only use vertices from the upper plane
                // to prevent double-counting, and set z=0 manually
                if (box.is2D())
                {
                    if (vert_z < 0)
                    {
                        continue;
                    }
                    vert_z = 0;
                }
                vec3<double> delta = vec3<double>(vert_x, vert_y, vert_z) - query_point;
                relative_vertices.push_back(delta);
            }

            // Sort relative vertices by their angle in 2D systems
            if (box.is2D())
            {
                std::sort(relative_vertices.begin(), relative_vertices.end(),
                          [](const vec3<double>& a, const vec3<double>& b) {
                              return std::atan2(a.y, a.x) < std::atan2(b.y, b.x);
                          });
            }

            // Save polytope vertices in system coordinates
            const vec3<double>& query_point_system_coords((*nq)[query_point_id]);

            std::vector<vec3<double>> system_vertices;
            system_vertices.reserve(relative_vertices.size());
            std::transform(
                relative_vertices.begin(), relative_vertices.end(), std::back_inserter(system_vertices),
                [&](const auto& relative_vertex) { return relative_vertex + query_point_system_coords; });
            m_polytopes[query_point_id] = system_vertices;

            // Save cell volume
            m_volumes[query_point_id] = cell.volume();

            // Compute cell neighbors
            size_t neighbor_counter(0);
            size_t face_vertices_index(0);
            for (auto neighbor_iterator = neighbors.begin(); neighbor_iterator != neighbors.end();
                 neighbor_iterator++, neighbor_counter++, face_vertices_index += face_vertices[face_vertices_index] + 1)
            {
                // Get the normal to the current face
                const vec3<double> normal(normals[3 * neighbor_counter], normals[3 * neighbor_counter + 1],
                                          normals[3 * neighbor_counter + 2]);

                // Ignore bonds in 2D systems that point up or down. This check
                // should only be dealing with bonds whose normal vectors' z
                // components are -1, 0, or +1 (within some tolerance). This
                // also skips bonds where the normal vector is exactly zero.
                // A normal vector of exactly zero seems to appear for certain
                // particles in 2D systems where the neighbors are very close.
                // It seems like an issue of numerical imprecision but could be
                // some other pathological case.
                if (box.is2D() && std::abs(normal.z) > 0.5
                    || (normal.x == 0 && normal.y == 0 && normal.z == 0))
                {
                    continue;
                }

                // Fetch neighbor information
                const int point_id = *neighbor_iterator;
                const float weight(face_areas[neighbor_counter]);

                // Find a vertex on the current face: this leverages the
                // structure of face_vertices, which has a count of the
                // number of vertices for a face followed by the
                // corresponding vertex ids for that face. We use this
                // structure later when incrementing face_vertices_index.
                // face_vertices_index always points to the "vertex
                // counter" element of face_vertices for the current face.

                // Get the id of the vertex on this face that is most parallel to the normal
                const auto normal_length = std::sqrt(dot(normal, normal));
                auto cosine_vertex_to_normal = [&](const auto& vertex_id_on_face){
                    const vec3<double> rv(vertices[3 * vertex_id_on_face], vertices[3 * vertex_id_on_face + 1],
                                          vertices[3 * vertex_id_on_face + 2]);
                    const vec3<double> riv(rv - query_point);
                    return dot(riv, normal) / std::sqrt(dot(riv, riv)) / normal_length;
                };

                const int vertex_id_on_face = *std::max_element(
                    &(face_vertices[face_vertices_index + 1]),
                    &(face_vertices[face_vertices_index + face_vertices[face_vertices_index]]),
                    [&](const auto& a, const auto& b){
                        return cosine_vertex_to_normal(a) < cosine_vertex_to_normal(b);
                    });

                // Project the vertex vector onto the face normal to get a
                // vector from query_point to the face, then double it to
                // get the vector to the neighbor particle.
                const vec3<double> rv(vertices[3 * vertex_id_on_face], vertices[3 * vertex_id_on_face + 1],
                                      vertices[3 * vertex_id_on_face + 2]);
                const vec3<double> riv(rv - query_point);
                const vec3<float> vector(2.0 * dot(riv, normal) * normal);


                // Compute the distance from query_point to point in the direction of the normal.
                const float distance(std::sqrt(dot(vector, vector)));

                bonds.emplace_back(query_point_id, point_id, distance, weight, vector);
            }

        } while (voronoi_loop.inc());
    }

    tbb::parallel_sort(bonds.begin(), bonds.end(), [](const NeighborBond& n1, const NeighborBond& n2) {
        return n1.less_id_ref_weight(n2);
    });

    unsigned int num_bonds = bonds.size();

    m_neighbor_list->resize(num_bonds);
    m_neighbor_list->setNumBonds(num_bonds, n_points, n_points);

    util::forLoopWrapper(0, num_bonds, [=](size_t begin, size_t end) {
        for (size_t bond = begin; bond != end; ++bond)
        {
            m_neighbor_list->getNeighbors()(bond, 0) = bonds[bond].query_point_idx;
            m_neighbor_list->getNeighbors()(bond, 1) = bonds[bond].point_idx;
            m_neighbor_list->getDistances()[bond] = bonds[bond].distance;
            m_neighbor_list->getWeights()[bond] = bonds[bond].weight;
            m_neighbor_list->getVectors()[bond] = bonds[bond].vector;
        }
    });
}

}; }; // end namespace freud::locality
