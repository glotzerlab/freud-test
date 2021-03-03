// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFTR12_H
#define PMFTR12_H

#include "PMFT.h"

/*! \file PMFTR12.h
    \brief Routines for computing potential of mean force and torque in R12 coordinates
*/

namespace freud { namespace pmft {

class PMFTR12 : public PMFT
{
public:
    //! Constructor
    PMFTR12(float r_max, unsigned int n_r, unsigned int n_t1, unsigned int n_t2);

    /*! Compute the PCF for the passed in set of points. The function will be added to previous values
        of the PCF
    */
    void accumulate(const locality::NeighborQuery* neighbor_query, const float* orientations,
                    const vec3<float>* query_points, const float* query_orientations,
                    unsigned int n_query_points, const locality::NeighborList* nlist,
                    freud::locality::QueryArgs qargs);

protected:
    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    void reduce() override;

    util::ManagedArray<float> m_inv_jacobian_array; //!< Array of inverse jacobians for each bin
};

}; }; // end namespace freud::pmft

#endif // PMFTR12_H
