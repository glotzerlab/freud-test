// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef PMFT_H
#define PMFT_H

#include <tbb/tbb.h>

#include "BondHistogramCompute.h"
#include "Box.h"
#include "Histogram.h"
#include "ManagedArray.h"
#include "VectorMath.h"

/*! \internal
    \file PMFT.h
    \brief Declares base class for all PMFT classes
*/

namespace freud { namespace pmft {

//! Computes the PMFT for a given set of points
/*! The PMFT class is an abstract class providing the basis for all classes calculating PMFTs for specific
 *  dimensional cases. The PMFT class defines some of the key interfaces required for all PMFT classes, such
 *  as the ability to access the underlying PCF and box. Many of the specific methods must be implemented by
 *  subclasses that account for the proper set of dimensions.The required functions are implemented as pure
 *  virtual functions here to enforce this.
 */
class PMFT : public locality::BondHistogramCompute
{
public:
    //! Constructor
    PMFT() : BondHistogramCompute() {}

    //! Destructor
    ~PMFT() override = default;

    //! Get a reference to the PCF array
    const util::ManagedArray<float>& getPCF()
    {
        return reduceAndReturn(m_pcf_array);
    }

protected:
    //! Reduce the thread local histogram into the total pair correlation function.
    /*! The pair correlation function is computed by reducing the bin counts in
     * all of the thread local histograms and then multiplying these by the
     * appropriate normalization factor. The normalization is nearly identical
     * to the RDF except for the volume normalization. In the RDF, the volume
     * normalization is simply V_shell/V_total, but in the PFMT the
     * normalization depends on the volume element in the relevant coordinate
     * system. This method accepts a function jf that returns the volume
     * element corresponding to a given bin in the histogram.
     *
     *  **IMPORTANT NOTE**: The inv_num_dens factor in the calculation in this
     *  function is just volume / Np, so it does not include volume elements in
     *  the orientational degrees of freedom. This means that the corresponding
     *  normalization factors *should not* be applied to the Jacobian factor
     *  argument. For instance, any full-dimensional PMFT in 2D must contain at
     *  least one angular term, but that term should not contain a factor of
     *  2*PI since that factor is effectively divided out of the volume here.
     *
     *  \param JacobFactor A function with one parameter (the histogram bin index) that returns the volume of
     * the element in the histogram bin corresponding to the index.
     */
    template<typename JacobFactor> void reduce(JacobFactor jf)
    {
        m_pcf_array.prepare(m_histogram.shape());
        m_histogram.prepare(m_histogram.shape());

        float inv_num_dens = m_box.getVolume() / static_cast<float>(m_n_query_points);
        float norm_factor
            = float(1.0) / (static_cast<float>(m_frame_counter) * static_cast<float>(m_n_points));
        float prefactor = inv_num_dens * norm_factor;

        m_histogram.reduceOverThreadsPerBin(m_local_histograms, [this, &prefactor, &jf](size_t i) {
            m_pcf_array[i] = m_histogram[i] * prefactor * jf(i);
        });
    }

    util::ManagedArray<float> m_pcf_array; //!< Array of computed pair correlation function.
};

}; }; // end namespace freud::pmft

#endif // PMFT_H
