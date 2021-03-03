// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef SPHERE_VOXELIZATION_H
#define SPHERE_VOXELIZATION_H

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file SphereVoxelization.h
    \brief Computes a grid of voxels occupied by spheres.
*/

namespace freud { namespace density {

//! Computes a grid of voxels occupied by spheres.
/*! This class constructs a grid of voxels. From a given set of points and a
    desired radius, a set of spheres are created. The voxels are assigned a
    value of 1 if their center is contained in one or more spheres and 0
    otherwise. The dimensions of the grid are set in the constructor, and can
    either be set equally for all dimensions or for each dimension
    independently.
*/
class SphereVoxelization
{
public:
    //! Constructor
    SphereVoxelization(vec3<unsigned int> width, float r_max);

    // Destructor
    ~SphereVoxelization() = default;

    //! Get the simulation box.
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Get sphere radius used for voxelization.
    float getRMax() const
    {
        return m_r_max;
    }

    //! Compute the voxelization.
    void compute(const freud::locality::NeighborQuery* nq);

    //! Get a reference to the last computed voxels.
    const util::ManagedArray<unsigned int>& getVoxels() const;

    vec3<unsigned int> getWidth() const;

private:
    box::Box m_box;             //!< Simulation box containing the points.
    vec3<unsigned int> m_width; //!< Number of bins in the grid in each dimension.
    float m_r_max;              //!< Sphere radius used for voxelization.
    bool m_has_computed;        //!< Tracks whether a call to compute has been made.

    util::ManagedArray<unsigned int> m_voxels_array; //! Computed voxels array.
};

}; }; // end namespace freud::density

#endif // SPHERE_VOXELIZATION_H
