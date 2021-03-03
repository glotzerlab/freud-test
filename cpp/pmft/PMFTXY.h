// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFTXY_H
#define PMFTXY_H

#include "PMFT.h"

/*! \file PMFTXY.h
    \brief Routines for computing 2D potential of mean force in XY coordinates
*/

namespace freud { namespace pmft {

class PMFTXY : public PMFT
{
public:
    //! Constructor
    PMFTXY(float x_max, float y_max, unsigned int n_x, unsigned int n_y);

    /*! Compute the PCF for the passed in set of points. The result will
     *  be added to previous values of the PCF.
     */
    void accumulate(const locality::NeighborQuery* neighbor_query, const float* query_orientations,
                    const vec3<float>* query_points, unsigned int n_query_points,
                    const locality::NeighborList* nlist, freud::locality::QueryArgs qargs);

protected:
    //! \internal
    //! helper function to reduce the thread specific arrays into one array
    void reduce() override;

    float m_jacobian; //!< Determinant of Jacobian, bin area
};

}; }; // end namespace freud::pmft

#endif // PMFTXY_H
