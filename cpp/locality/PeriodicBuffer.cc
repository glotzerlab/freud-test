// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <stdexcept>

#include "PeriodicBuffer.h"

/*! \file PeriodicBuffer.cc
    \brief Replicates points across periodic boundaries.
*/

namespace freud { namespace locality {

void PeriodicBuffer::compute(const freud::locality::NeighborQuery* neighbor_query, const vec3<float>& buff,
                             const bool use_images)
{
    m_box = neighbor_query->getBox();
    if (buff.x < 0)
    {
        throw std::invalid_argument("Buffer x distance must be non-negative.");
    }
    if (buff.y < 0)
    {
        throw std::invalid_argument("Buffer y distance must be non-negative.");
    }
    if (buff.z < 0)
    {
        throw std::invalid_argument("Buffer z distance must be non-negative.");
    }

    // Get the box dimensions
    vec3<float> L(m_box.getL());
    float xy = m_box.getTiltFactorXY();
    float xz = m_box.getTiltFactorXZ();
    float yz = m_box.getTiltFactorYZ();
    bool is2D = m_box.is2D();
    vec3<int> images;

    if (use_images)
    {
        images = vec3<int>(std::ceil(buff.x), std::ceil(buff.y), std::ceil(buff.z));
        m_buffer_box
            = freud::box::Box(static_cast<float>(1 + images.x) * L.x, static_cast<float>(1 + images.y) * L.y,
                              static_cast<float>(1 + images.z) * L.z, xy, xz, yz, is2D);
    }
    else
    {
        images = vec3<int>(std::ceil(buff.x / L.x), std::ceil(buff.y / L.y), std::ceil(buff.z / L.z));
        m_buffer_box
            = freud::box::Box(L.x + 2 * buff.x, L.y + 2 * buff.y, L.z + 2 * buff.z, xy, xz, yz, is2D);
    }

    if (is2D)
    {
        images.z = 0;
    }

    m_buffer_points.clear();
    m_buffer_ids.clear();

    // for each point
    for (unsigned int point_id = 0; point_id < neighbor_query->getNPoints(); point_id++)
    {
        for (int i = use_images ? 0 : -images.x; i <= images.x; i++)
        {
            for (int j = use_images ? 0 : -images.y; j <= images.y; j++)
            {
                for (int k = use_images ? 0 : -images.z; k <= images.z; k++)
                {
                    // Skip the origin image
                    if (i == 0 && j == 0 && k == 0)
                    {
                        continue;
                    }

                    // Compute the new position for the buffer point,
                    // shifted by images.
                    vec3<float> point_image = (*neighbor_query)[point_id];
                    point_image += float(i) * m_box.getLatticeVector(0);
                    point_image += float(j) * m_box.getLatticeVector(1);
                    if (!is2D)
                    {
                        point_image += float(k) * m_box.getLatticeVector(2);
                    }

                    if (use_images)
                    {
                        // Wrap the positions back into the buffer box and
                        // always append them if a number of images was
                        // specified. Performing the check this way ensures we
                        // have the correct number of points instead of
                        // relying on the floating point precision of the
                        // fractional check below.
                        m_buffer_points.push_back(m_buffer_box.wrap(point_image));
                        m_buffer_ids.push_back(point_id);
                    }
                    else
                    {
                        // When using a buffer "skin distance," we check the
                        // fractional coordinates to see if the points are
                        // inside the buffer box. Unexpected results may occur
                        // due to numerical imprecision in this check!
                        vec3<float> buff_frac = m_buffer_box.makeFractional(point_image);
                        if (0 <= buff_frac.x && buff_frac.x < 1 && 0 <= buff_frac.y && buff_frac.y < 1
                            && (is2D || (0 <= buff_frac.z && buff_frac.z < 1)))
                        {
                            m_buffer_points.push_back(point_image);
                            m_buffer_ids.push_back(point_id);
                        }
                    }
                }
            }
        }
    }
}

}; }; // end namespace freud::locality
