// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef RDF_H
#define RDF_H

#include "BondHistogramCompute.h"
#include "Box.h"
#include "Histogram.h"

/*! \file RDF.h
    \brief Routines for computing radial density functions.
*/

namespace freud { namespace density {
class RDF : public locality::BondHistogramCompute
{
public:
    //! Constructor
    RDF(unsigned int bins, float r_max, float r_min = 0, bool normalize = false);

    //! Destructor
    ~RDF() override = default;

    //! Compute the RDF
    /*! Accumulate the given points to the histogram. Accumulation is performed
     * in parallel on thread-local copies of the data, which are reduced into
     * the primary data arrays when the user requests outputs.
     */
    void accumulate(const freud::locality::NeighborQuery* neighbor_query, const vec3<float>* query_points,
                    unsigned int n_query_points, const freud::locality::NeighborList* nlist,
                    freud::locality::QueryArgs qargs);

    //! Reduce thread-local arrays onto the primary data arrays.
    void reduce() override;

    //! Get the positional correlation function.
    const util::ManagedArray<float>& getRDF()
    {
        return reduceAndReturn(m_pcf);
    }

    //! Get a reference to the N_r array.
    /*! Mathematically, m_N_r[i] is the average number of points
     * contained within a ball of radius getBinEdges()[i+1] centered at a given
     * query_point, averaged over all query_points.
     */
    const util::ManagedArray<float>& getNr()
    {
        return reduceAndReturn(m_N_r);
    }

private:
    bool m_normalize;                //!< Whether to enforce that the RDF should tend to 1 (instead of
                                     //!< num_query_points/num_points).
    util::ManagedArray<float> m_pcf; //!< The computed pair correlation function.
    util::ManagedArray<float>
        m_N_r; //!< Cumulative bin sum N(r) (the average number of points in a ball of radius r).
    util::ManagedArray<float>
        m_vol_array2D; //!< Areas of concentric rings corresponding to the histogram bins in 2D.
    util::ManagedArray<float>
        m_vol_array3D; //!< Areas of concentric spherical shells corresponding to the histogram bins in 3D.
};

}; }; // end namespace freud::density

#endif // RDF_H
