// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is part of the freud project, released under the BSD 3-Clause License.

#ifndef LOCAL_BOND_PROJECTION_H
#define LOCAL_BOND_PROJECTION_H

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file LocalBondProjection.h
    \brief Compute the projection of nearest neighbor bonds for each particle onto some
    set of reference vectors, defined in the particles' local reference frame.
*/

namespace freud { namespace environment {

//! Project the local bond onto all symmetrically equivalent vectors to proj_vec.
//! Return the maximal projection value.
float computeMaxProjection(const vec3<float>& proj_vec, const vec3<float>& local_bond,
                           const quat<float>* equiv_qs, unsigned int n_equiv_qs);

class LocalBondProjection
{
public:
    //! Constructor
    LocalBondProjection() = default;

    //! Destructor
    ~LocalBondProjection() = default;

    //! Compute the maximal local bond projection
    void compute(const locality::NeighborQuery* nq, const quat<float>* orientations,
                 const vec3<float>* query_points, unsigned int n_query_points, const vec3<float>* proj_vecs,
                 unsigned int n_proj, const quat<float>* equiv_orientations,
                 unsigned int n_equiv_orientations, const freud::locality::NeighborList* nlist,
                 locality::QueryArgs qargs);

    //! Get a reference to the last computed maximal local bond projection array
    const util::ManagedArray<float>& getProjections() const
    {
        return m_local_bond_proj;
    }

    //! Get a reference to the last computed normalized maximal local bond projection array
    const util::ManagedArray<float>& getNormedProjections() const
    {
        return m_local_bond_proj_norm;
    }

    //! Return a pointer to the NeighborList used in the last call to compute.
    locality::NeighborList* getNList()
    {
        return &m_nlist;
    }

private:
    locality::NeighborList m_nlist; //!< The NeighborList used in the last call to compute.

    util::ManagedArray<float> m_local_bond_proj;      //!< Local bond projection array computed
    util::ManagedArray<float> m_local_bond_proj_norm; //!< Normalized local bond projection array computed
};

}; }; // end namespace freud::environment

#endif // LOCAL_BOND_PROJECTION_H
