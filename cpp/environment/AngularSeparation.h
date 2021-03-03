// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef ANGULAR_SEPARATION_H
#define ANGULAR_SEPARATION_H

#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file AngularSeparation.h
    \brief Compute the angular separations.
*/

namespace freud { namespace environment {

//! Compute the angular separation for a set of points
/*! Given a set of global orientations, this method accepts a set of
 * orientations that are compared against the global orientations to determine
 * the total angular distance between them. The output is an array of shape
 * (num_orientations, num_global_orientations) containing the pairwise
 * separation angles between the provided orientations and global orientations.
 */
class AngularSeparationGlobal
{
public:
    //! Constructor
    AngularSeparationGlobal() = default;

    //! Destructor
    ~AngularSeparationGlobal() = default;

    //! Compute the angular separation with respect to global orientation
    void compute(const quat<float>* global_orientations, unsigned int n_global,
                 const quat<float>* orientations, unsigned int n_points,
                 const quat<float>* equiv_orientations, unsigned int n_equiv_orientations);

    //! Returns the last computed global angle array
    const util::ManagedArray<float>& getAngles() const
    {
        return m_angles;
    }

private:
    util::ManagedArray<float> m_angles; //!< Global angle array computed
};

//! Compute the difference in orientation between pairs of points.
/*! Given two sets of oriented points and the bonds between these points, this
 * class computes the minimum separating angle between the orientations of each
 * pair of bonded points.
 */
class AngularSeparationNeighbor
{
public:
    //! Constructor
    AngularSeparationNeighbor() = default;

    //! Destructor
    ~AngularSeparationNeighbor() = default;

    //! Compute the angular separation between neighbors
    void compute(const locality::NeighborQuery* nq, const quat<float>* orientations,
                 const vec3<float>* query_points, const quat<float>* query_orientations,
                 unsigned int n_query_points, const quat<float>* equiv_orientations,
                 unsigned int n_equiv_orientations, const freud::locality::NeighborList* nlist,
                 locality::QueryArgs qargs);

    //! Returns the last computed neighbor angle array
    const util::ManagedArray<float>& getAngles() const
    {
        return m_angles;
    }

    //! Return a pointer to the NeighborList used in the last call to compute.
    locality::NeighborList* getNList()
    {
        return &m_nlist;
    }

private:
    util::ManagedArray<float> m_angles; //!< neighbor angle array computed
    locality::NeighborList m_nlist;     //!< The NeighborList used in the last call to compute.
};

}; }; // end namespace freud::environment

#endif // ANGULAR_SEPARATION_H
