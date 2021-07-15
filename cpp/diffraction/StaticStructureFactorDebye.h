// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STATIC_STRUCTURE_FACTOR_H
#define STATIC_STRUCTURE_FACTOR_H

#include "Box.h"
#include "Histogram.h"
#include "NeighborQuery.h"

/*! \file StaticStructureFactorDebye.h
    \brief Routines for computing static structure factors.

    Computes structure factors from the Fourier transform of a radial distribution function (RDF).
    This method is not capable of resolving k-vectors smaller than the magnitude 4 * pi / L, where L is the
    smallest side length of the system's periodic box.
*/

namespace freud { namespace diffraction {

class StaticStructureFactorDebye
{
    using StaticStructureFactorDebyeHistogram = util::Histogram<float>;

public:
    //! Constructor
    StaticStructureFactorDebye(unsigned int bins, float k_max, float k_min = 0);

    //! Destructor
    virtual ~StaticStructureFactorDebye() {};

    //! Compute the structure factor S(k) using the Debye formula
    void accumulate(const freud::locality::NeighborQuery* neighbor_query, const vec3<float>* query_points,
                    unsigned int n_query_points);

    //! Reduce thread-local arrays onto the primary data arrays.
    void reduce();

    //! Return thing_to_return after reducing if necessary.
    template<typename U> U& reduceAndReturn(U& thing_to_return)
    {
        if (m_reduce == true)
        {
            reduce();
        }
        m_reduce = false;
        return thing_to_return;
    }

    //! Reset the histogram to all zeros
    void reset()
    {
        m_local_histograms.reset();
        m_frame_counter = 0;
        m_reduce = true;
    }

    //! Get the structure factor
    const util::ManagedArray<float>& getStructureFactor()
    {
        return reduceAndReturn(m_structure_factor);
    }

    //! Get the k bin edges
    const std::vector<float> getBinEdges() const
    {
        return m_histogram.getBinEdges()[0];
    }

    //! Get the k bin centers
    const std::vector<float> getBinCenters() const
    {
        return m_histogram.getBinCenters()[0];
    }

    //! Get the minimum valid k value
    float getMinValidK() const
    {
        return m_min_valid_k;
    }

private:
    StaticStructureFactorDebyeHistogram m_histogram; //!< Histogram to hold computed structure factor
    StaticStructureFactorDebyeHistogram::ThreadLocalHistogram
        m_local_histograms;                       //!< Thread local histograms for TBB parallelism
    util::ManagedArray<float> m_structure_factor; //!< The computed structure factor
    unsigned int m_frame_counter;                 //!< Number of frames calculated.
    float m_min_valid_k;                          //!< The minimum valid k-vector based on the computed box
    bool m_reduce;                                //!< Whether to reduce
};

}; }; // namespace freud::diffraction

#endif // STATIC_STRUCTURE_FACTOR_H
