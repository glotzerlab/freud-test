// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef HEXATIC_TRANSLATIONAL_H
#define HEXATIC_TRANSLATIONAL_H

#include <complex>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborComputeFunctional.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"

/*! \file HexaticTranslational.h
    \brief Compute the hexatic/translational order parameter for each particle.
*/

namespace freud { namespace order {

//! Parent class for Hexatic and Translational
/*!
 */
template<typename T> class HexaticTranslational
{
public:
    //! Constructor
    explicit HexaticTranslational(T k, bool weighted = false) : m_k(k), m_weighted(weighted) {}

    //! Destructor
    virtual ~HexaticTranslational() = default;

    T getK() const
    {
        return m_k;
    }

    bool isWeighted() const
    {
        return m_weighted;
    }

    //! Get a reference to the order parameter array
    const util::ManagedArray<std::complex<float>>& getOrder() const
    {
        return m_psi_array;
    }

protected:
    //! Compute the order parameter
    template<typename Func>
    void computeGeneral(Func func, const freud::locality::NeighborList* nlist,
                        const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs,
                        bool normalize_by_k);

    const T m_k; //!< The symmetry order for Hexatic, or normalization for Translational
    const bool
        m_weighted; //!< Whether to use neighbor weights in computing the order parameter (default false)
    util::ManagedArray<std::complex<float>> m_psi_array; //!< psi array computed
};

//! Compute the hexatic order parameter for a set of points
/*!
 */
class Hexatic : public HexaticTranslational<unsigned int>
{
public:
    //! Constructor
    Hexatic(unsigned int k = 6, bool weighted = false);

    //! Destructor
    ~Hexatic() override = default;

    //! Compute the hexatic order parameter
    void compute(const freud::locality::NeighborList* nlist, const freud::locality::NeighborQuery* points,
                 freud::locality::QueryArgs qargs);
};

//! Compute the translational order parameter for a set of points
/*! THIS CLASS IS DEPRECATED AND WILL BE REMOVED IN THE NEXT MAJOR RELEASE OF FREUD.
 */
class Translational : public HexaticTranslational<float>
{
public:
    //! Constructor
    Translational(float k = 6, bool weighted = false);

    //! Destructor
    ~Translational() override = default;

    //! Compute the translational order parameter
    void compute(const freud::locality::NeighborList* nlist, const freud::locality::NeighborQuery* points,
                 freud::locality::QueryArgs qargs);
};

}; }; // end namespace freud::order

#endif // HEXATIC_TRANSLATIONAL_H
