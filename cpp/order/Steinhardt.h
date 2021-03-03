// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef STEINHARDT_H
#define STEINHARDT_H

#include <complex>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "ThreadStorage.h"
#include "VectorMath.h"
#include "Wigner3j.h"
#include "fsph/src/spherical_harmonics.hpp"

/*! \file Steinhardt.h
    \brief Computes variants of Steinhardt order parameters.
*/

namespace freud { namespace order {

//! Compute the Steinhardt local rotationally invariant ql or wl order parameter for a set of points
/*!
 * Implements the rotationally invariant ql or wl order parameter described
 * by Steinhardt. For a particle i, we calculate the average Q_l by summing
 * the spherical harmonics between particle i and its neighbors j in a local
 * region:
 * \f$ \overline{Q}_{lm}(i) = \frac{1}{N_b} \displaystyle\sum_{j=1}^{N_b}
 * Y_{lm}(\theta(\vec{r}_{ij}),\phi(\vec{r}_{ij})) \f$
 *
 * This is then combined in a rotationally invariant fashion to remove local
 * orientational order as follows:
 * \f$ Q_l(i)=\sqrt{\frac{4\pi}{2l+1} \displaystyle\sum_{m=-l}^{l} |\overline{Q}_{lm}|^2 }  \f$
 *
 * If the average flag is set, the order parameters averages over the second neighbor shell.
 * For a particle i, we calculate the average Q_l by summing the spherical
 * harmonics between particle i and its neighbors j and the neighbors k of
 * neighbor j in a local region.
 *
 * If the norm flag is set, the ql value is normalized by the average qlm value
 * for the system.
 *
 * If the flag wl is set, the third-order invariant wl order parameter will
 * be calculated. wl can aid in distinguishing between FCC, HCP, and BCC.
 *
 * If the flag wl_normalize is set, the third-order invariant wl order parameter
 * will be normalized.
 *
 * For more details see:
 * - PJ Steinhardt (1983) (DOI: 10.1103/PhysRevB.28.784)
 * - Wolfgang Lechner (2008) (DOI: 10.1063/Journal of Chemical Physics 129.114707)
 */

class Steinhardt
{
public:
    //! Steinhardt Class Constructor
    /*! Constructor for Steinhardt analysis class.
     *  \param l Spherical harmonic number l.
     *           Must be a positive number.
     */
    explicit Steinhardt(unsigned int l, bool average = false, bool wl = false, bool weighted = false,
                        bool wl_normalize = false)
        : m_l(l), m_num_ms(2 * l + 1), m_average(average), m_wl(wl), m_weighted(weighted),
          m_wl_normalize(wl_normalize), m_qlm_local(2 * l + 1)

    {}

    //! Empty destructor
    ~Steinhardt() = default;

    //! Get the number of particles used in the last compute
    unsigned int getNP() const
    {
        return m_Np;
    }

    //! Get the last calculated order parameter
    const util::ManagedArray<float>& getParticleOrder() const
    {
        if (m_wl)
        {
            return m_wli;
        }
        return getQl();
    }

    //! Get the last calculated ql
    const util::ManagedArray<float>& getQl() const
    {
        if (m_average)
        {
            return m_qliAve;
        }
        return m_qli;
    }

    //! Get the last calculated qlm for each particle
    const util::ManagedArray<std::complex<float>>& getQlm() const
    {
        return m_qlmi;
    }

    //! Get system-normalized order
    float getOrder() const
    {
        return m_norm;
    }

    //!< Whether to take a second shell average
    bool isAverage() const
    {
        return m_average;
    }

    //!< Whether to use the third-order invariant wl
    bool isWl() const
    {
        return m_wl;
    }

    //!< Whether to use neighbor weights in computing qlmi
    bool isWeighted() const
    {
        return m_weighted;
    }

    //!< Whether to normalize the third-order invariant wl
    bool isWlNormalized() const
    {
        return m_wl_normalize;
    }

    //! Compute the order parameter
    void compute(const freud::locality::NeighborList* nlist, const freud::locality::NeighborQuery* points,
                 freud::locality::QueryArgs qargs);

    unsigned int getL() const
    {
        return m_l;
    }

private:
    //! \internal
    //! Spherical harmonics calculation for Ylm filling a
    //  std::vector<std::complex<float> > with values for m = -l..l.
    void computeYlm(const float theta, const float phi, std::vector<std::complex<float>>& Ylm) const;

    template<typename T> std::shared_ptr<T> makeArray(size_t size);

    //! Reallocates only the necessary arrays when the number of particles changes
    // unsigned int Np number of particles
    void reallocateArrays(unsigned int Np);

    //! Calculates qlms and the ql order parameter before any further modifications
    void baseCompute(const freud::locality::NeighborList* nlist, const freud::locality::NeighborQuery* points,
                     freud::locality::QueryArgs qargs);

    //! Calculates the neighbor average ql order parameter
    void computeAve(const freud::locality::NeighborList* nlist, const freud::locality::NeighborQuery* points,
                    freud::locality::QueryArgs qargs);

    //! Compute the system-wide order by averaging over particles, then
    //  reducing over the m values to produce a single scalar.
    float normalizeSystem();

    //! Sum over Wigner 3j coefficients to compute third-order invariants
    //  wl from second-order invariants ql
    void aggregatewl(util::ManagedArray<float>& target, const util::ManagedArray<std::complex<float>>& source,
                     const util::ManagedArray<float>& normalization_source) const;

    // Member variables used for compute
    unsigned int m_Np {0}; //!< Last number of points computed
    unsigned int m_l;      //!< Spherical harmonic l value.
    unsigned int m_num_ms; //!< The number of magnetic quantum numbers (2*m_l+1).

    // Flags
    bool m_average;      //!< Whether to take a second shell average (default false)
    bool m_wl;           //!< Whether to use the third-order invariant wl (default false)
    bool m_weighted;     //!< Whether to use neighbor weights in computing qlmi (default false)
    bool m_wl_normalize; //!< Whether to normalize the third-order invariant wl (default false)

    util::ManagedArray<std::complex<float>> m_qlmi;       //!< qlm for each particle i
    util::ManagedArray<std::complex<float>> m_qlm;        //!< Normalized qlm(Ave) for the whole system
    util::ThreadStorage<std::complex<float>> m_qlm_local; //!< Thread-specific m_qlm(Ave)
    util::ManagedArray<float> m_qli;    //!< ql locally invariant order parameter for each particle i
    util::ManagedArray<float> m_qliAve; //!< Averaged ql with 2nd neighbor shell for each particle i
    util::ManagedArray<std::complex<float>>
        m_qlmiAve; //!< Averaged qlm with 2nd neighbor shell for each particle i
    util::ManagedArray<std::complex<float>> m_qlmAve; //!< Normalized qlmiAve for the whole system
    float m_norm {0};                                 //!< System normalized order parameter
    util::ManagedArray<float>
        m_wli; //!< wl order parameter for each particle i, also used for wl averaged data
};

}; };  // end namespace freud::order
#endif // STEINHARDT_H
