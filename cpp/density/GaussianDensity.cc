// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cmath>
#include <stdexcept>

#include "GaussianDensity.h"

/*! \file GaussianDensity.cc
    \brief Routines for computing Gaussian smeared densities from points.
*/

namespace freud { namespace density {

GaussianDensity::GaussianDensity(vec3<unsigned int> width, float r_max, float sigma)
    : m_box(), m_width(width), m_r_max(r_max), m_sigma(sigma), m_has_computed(false)
{
    if (r_max <= 0)
    {
        throw std::invalid_argument("GaussianDensity requires r_max to be positive.");
    }
}

//! Get a reference to the last computed Density
const util::ManagedArray<float>& GaussianDensity::getDensity() const
{
    return m_density_array;
}

//! Get width.
vec3<unsigned int> GaussianDensity::getWidth()
{
    return m_width;
}

//! Compute the density array.
void GaussianDensity::compute(const freud::locality::NeighborQuery* nq, const float* values)
{
    // set the number of dimensions for the calculation the first time it is done
    if (!m_has_computed || nq->getBox().is2D() == m_box.is2D())
    {
        m_box = nq->getBox();
        m_has_computed = true;
    }
    else
    {
        throw std::invalid_argument("The dimensionality of the box passed to GaussianDensity has "
                                    "changed. A new instance must be created to handle a different "
                                    "number of dimensions.");
    }

    auto n_points = nq->getNPoints();

    // if the user gives a single number for width, but the nq box is 2D, and
    // we want a 2D calculation
    if (m_box.is2D())
    {
        m_width.z = 1;
    }

    m_density_array.prepare({m_width.x, m_width.y, m_width.z});
    util::ThreadStorage<float> local_bin_counts({m_width.x, m_width.y, m_width.z});

    // set up some constants first
    const float Lx = m_box.getLx();
    const float Ly = m_box.getLy();
    const float Lz = m_box.getLz();
    const vec3<bool> periodic = m_box.getPeriodic();

    const float grid_size_x = Lx / m_width.x;
    const float grid_size_y = Ly / m_width.y;
    const float grid_size_z = m_box.is2D() ? 0 : Lz / m_width.z;

    // Find the number of bins within r_max
    const int bin_cut_x = int(m_r_max / grid_size_x);
    const int bin_cut_y = int(m_r_max / grid_size_y);
    const int bin_cut_z = m_box.is2D() ? 0 : int(m_r_max / grid_size_z);
    const float r_max_sq = m_r_max * m_r_max;
    const float sigmasq = m_sigma * m_sigma;
    const float normalization_base = float(1.0) / std::sqrt(constants::TWO_PI * sigmasq);
    const float dimensions = m_box.is2D() ? float(2.0) : float(3.0);
    const float normalization = std::pow(normalization_base, dimensions);

    util::forLoopWrapper(0, n_points, [&](size_t begin, size_t end) {
        // for each reference point
        for (size_t idx = begin; idx < end; ++idx)
        {
            const vec3<float> point = (*nq)[idx];
            const float value = (values != nullptr) ? values[idx] : 1.0f;

            // Find which bin the particle is in
            int bin_x = int((point.x + Lx / float(2.0)) / grid_size_x);
            int bin_y = int((point.y + Ly / float(2.0)) / grid_size_y);
            int bin_z = int((point.z + Lz / float(2.0)) / grid_size_z);

            // In 2D, only loop over the z=0 plane
            if (m_box.is2D())
            {
                bin_z = 0;
            }

            // Reject bins that are outside the box in aperiodic directions
            // Only evaluate over bins that are within the cutoff
            for (int k = bin_z - bin_cut_z; k <= bin_z + bin_cut_z; k++)
            {
                if (!periodic.z && (k < 0 || k >= int(m_width.z)))
                {
                    continue;
                }
                const float dz = (grid_size_z * static_cast<float>(k)) + (grid_size_z / float(2.0)) - point.z
                    - (Lz / float(2.0));

                for (int j = bin_y - bin_cut_y; j <= bin_y + bin_cut_y; j++)
                {
                    if (!periodic.y && (j < 0 || j >= int(m_width.y)))
                    {
                        continue;
                    }
                    const float dy = (grid_size_y * static_cast<float>(j)) + (grid_size_y / float(2.0))
                        - point.y - (Ly / float(2.0));

                    for (int i = bin_x - bin_cut_x; i <= bin_x + bin_cut_x; i++)
                    {
                        if (!periodic.x && (i < 0 || i >= int(m_width.x)))
                        {
                            continue;
                        }
                        const float dx = (grid_size_x * static_cast<float>(i)) + (grid_size_x / float(2.0))
                            - point.x - (Lx / float(2.0));

                        // Calculate the distance from the particle to the grid cell
                        const vec3<float> delta = m_box.wrap(vec3<float>(dx, dy, dz));

                        const float r_sq = dot(delta, delta);

                        // Check to see if this distance is within the specified r_max
                        if (r_sq < r_max_sq)
                        {
                            // Evaluate the gaussian
                            const float gaussian
                                = value * normalization * std::exp(-r_sq / (float(2.0) * sigmasq));

                            // Assure that out of range indices are corrected for storage
                            // in the array i.e. bin -1 is actually bin 29 for nbins = 30
                            const unsigned int ni = (i + m_width.x) % m_width.x;
                            const unsigned int nj = (j + m_width.y) % m_width.y;
                            const unsigned int nk = (k + m_width.z) % m_width.z;

                            // Store the gaussian contribution
                            local_bin_counts.local()(ni, nj, nk) += gaussian;
                        }
                    }
                }
            }
        }
    });

    // Parallel reduction over thread storage
    local_bin_counts.reduceInto(m_density_array);
}

}; }; // end namespace freud::density
