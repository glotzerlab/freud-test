// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef GAUSSIAN_DENSITY_H
#define GAUSSIAN_DENSITY_H

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"
#include "ThreadStorage.h"
#include "VectorMath.h"

/*! \file GaussianDensity.h
    \brief Routines for computing Gaussian smeared densities from points.
*/

namespace freud { namespace density {

//! Computes the density of a system on a grid.
/*! Replaces particle positions with a gaussian and calculates the
        contribution from the grid based upon the distance of the grid cell
        from the center of the Gaussian.
*/
class GaussianDensity
{
public:
    //! Constructor
    GaussianDensity(vec3<unsigned int> width, float r_max, float sigma);

    // Destructor
    ~GaussianDensity() = default;

    //! Get the simulation box.
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Get the width of the gaussian distributions.
    float getSigma() const
    {
        return m_sigma;
    }

    //! Return the cutoff distance.
    float getRMax() const
    {
        return m_r_max;
    }

    //! Compute the density.
    void compute(const freud::locality::NeighborQuery* nq, const float* values = nullptr);

    //! Get a reference to the last computed density.
    const util::ManagedArray<float>& getDensity() const;

    vec3<unsigned int> getWidth();

private:
    box::Box m_box;             //!< Simulation box containing the points.
    vec3<unsigned int> m_width; //!< Number of bins in the grid in each dimension.
    float m_r_max;              //!< Max distance at which to compute density.
    float m_sigma;              //!< Gaussian width sigma.
    bool m_has_computed;        //!< Tracks whether a call to compute has been made.

    util::ManagedArray<float> m_density_array; //! Computed density array.
};

}; }; // end namespace freud::density

#endif // GAUSSIAN_DENSITY_H
