// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "Steinhardt.h"
#include "NeighborComputeFunctional.h"
#include "utils.h"

/*! \file Steinhardt.cc
    \brief Computes variants of Steinhardt order parameters.
*/

namespace freud { namespace order {

// Calculating Ylm using fsph module
void Steinhardt::computeYlm(const float theta, const float phi, std::vector<std::complex<float>>& Ylm) const
{
    if (Ylm.size() != m_num_ms)
    {
        Ylm.resize(m_num_ms);
    }

    fsph::PointSPHEvaluator<float> sph_eval(m_l);

    unsigned int m_index(0);
    sph_eval.compute(theta, phi);

    for (typename fsph::PointSPHEvaluator<float>::iterator iter(sph_eval.begin_l(m_l, 0, true));
         iter != sph_eval.end(); ++iter)
    {
        // Manually add the Condon-Shortley phase, (-1)^m, to positive odd m
        float phase = 1;
        if (m_index <= m_l && m_index % 2 == 1)
        {
            phase = -1;
        }

        Ylm[m_index] = phase * (*iter);
        ++m_index;
    }
}

void Steinhardt::reallocateArrays(unsigned int Np)
{
    m_Np = Np;
    m_qlmi.prepare({Np, m_num_ms});
    m_qlm.prepare(m_num_ms);
    m_qli.prepare(Np);
    if (m_average)
    {
        m_qlmiAve.prepare({Np, m_num_ms});
        m_qliAve.prepare(Np);
    }
    if (m_wl)
    {
        m_wli.prepare(Np);
    }
}

void Steinhardt::compute(const freud::locality::NeighborList* nlist,
                         const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    // Allocate and zero out arrays as necessary.
    reallocateArrays(points->getNPoints());

    // Computes the base qlmi required for each specialized order parameter
    baseCompute(nlist, points, qargs);

    if (m_average)
    {
        computeAve(nlist, points, qargs);
    }

    // Reduce qlm
    m_qlm_local.reduceInto(m_qlm);

    if (m_wl)
    {
        if (m_average)
        {
            aggregatewl(m_wli, m_qlmiAve, m_qliAve);
        }
        else
        {
            aggregatewl(m_wli, m_qlmi, m_qli);
        }
    }
    m_norm = normalizeSystem();
}

void Steinhardt::baseCompute(const freud::locality::NeighborList* nlist,
                             const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    const auto normalizationfactor = float(4.0 * M_PI / m_num_ms);
    // For consistency, this reset is done here regardless of whether the array
    // is populated in baseCompute or computeAve.
    m_qlm_local.reset();
    freud::locality::loopOverNeighborsIterator(
        points, points->getPoints(), m_Np, qargs, nlist,
        [=](size_t i, const std::shared_ptr<freud::locality::NeighborPerPointIterator>& ppiter) {
            float total_weight(0);
            const vec3<float> ref((*points)[i]);
            for (freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next())
            {
                const vec3<float> delta = points->getBox().wrap((*points)[nb.point_idx] - ref);
                const float weight(m_weighted ? nb.weight : float(1.0));

                // phi is usually in range 0..2Pi, but
                // it only appears in Ylm as exp(im\phi),
                // so range -Pi..Pi will give same results.
                float phi = std::atan2(delta.y, delta.x); // -Pi..Pi

                // This value must be clamped in cases where the particles are
                // aligned along z, otherwise due to floating point error we
                // could get delta.z/nb.distance = -1-eps, which is outside the
                // valid range of std::acos.
                float theta = std::acos(util::clamp(delta.z / nb.distance, -1, 1)); // 0..Pi

                // If the points are directly on top of each other,
                // theta should be zero instead of nan.
                if (nb.distance == float(0))
                {
                    theta = 0;
                }

                std::vector<std::complex<float>> Ylm(m_num_ms);
                computeYlm(theta, phi, Ylm); // Fill up Ylm

                for (unsigned int k = 0; k < m_num_ms; ++k)
                {
                    m_qlmi({static_cast<unsigned int>(i), k}) += weight * Ylm[k];
                }
                total_weight += weight;
            } // End loop going over neighbor bonds

            // Normalize!
            for (unsigned int k = 0; k < m_num_ms; ++k)
            {
                // Cache the index for efficiency.
                const unsigned int index = m_qlmi.getIndex({static_cast<unsigned int>(i), k});
                m_qlmi[index] /= total_weight;
                // Add the norm, which is the (complex) squared magnitude
                m_qli[i] += norm(m_qlmi[index]);
                // This array gets populated by computeAve in the averaging case.
                if (!m_average)
                {
                    m_qlm_local.local()[k] += m_qlmi[index] / float(m_Np);
                }
            }
            m_qli[i] *= normalizationfactor;
            m_qli[i] = std::sqrt(m_qli[i]);
        });
}

void Steinhardt::computeAve(const freud::locality::NeighborList* nlist,
                            const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    std::shared_ptr<locality::NeighborQueryIterator> iter;
    if (nlist == nullptr)
    {
        iter = points->query(points->getPoints(), points->getNPoints(), qargs);
    }

    const auto normalizationfactor = float(4.0 * M_PI / m_num_ms);

    freud::locality::loopOverNeighborsIterator(
        points, points->getPoints(), m_Np, qargs, nlist,
        [=](size_t i, const std::shared_ptr<freud::locality::NeighborPerPointIterator>& ppiter) {
            unsigned int neighborcount(1);
            for (freud::locality::NeighborBond nb1 = ppiter->next(); !ppiter->end(); nb1 = ppiter->next())
            {
                // Since we need to find neighbors of neighbors, we need to add some extra logic here to
                // create the appropriate iterators.
                std::shared_ptr<freud::locality::NeighborPerPointIterator> ns_neighbors_iter;
                if (nlist != nullptr)
                {
                    ns_neighbors_iter
                        = std::make_shared<locality::NeighborListPerPointIterator>(nlist, nb1.point_idx);
                }
                else
                {
                    ns_neighbors_iter = iter->query(nb1.point_idx);
                }

                for (freud::locality::NeighborBond nb2 = ns_neighbors_iter->next(); !ns_neighbors_iter->end();
                     nb2 = ns_neighbors_iter->next())
                {
                    for (unsigned int k = 0; k < m_num_ms; ++k)
                    {
                        // Adding all the qlm of the neighbors. We use the
                        // vector function signature for indexing into the
                        // arrays for speed.
                        m_qlmiAve({static_cast<unsigned int>(i), k}) += m_qlmi({nb2.point_idx, k});
                    }
                    neighborcount++;
                } // End loop over particle neighbor's bonds
            }     // End loop over particle's bonds

            // Normalize!
            for (unsigned int k = 0; k < m_num_ms; ++k)
            {
                // Cache the index for efficiency.
                const unsigned int index = m_qlmiAve.getIndex({static_cast<unsigned int>(i), k});
                // Adding the qlm of the particle i itself
                m_qlmiAve[index] += m_qlmi[index];
                m_qlmiAve[index] /= static_cast<float>(neighborcount);
                m_qlm_local.local()[k] += m_qlmiAve[index] / float(m_Np);
                // Add the norm, which is the complex squared magnitude
                m_qliAve[i] += norm(m_qlmiAve[index]);
            }
            m_qliAve[i] *= normalizationfactor;
            m_qliAve[i] = std::sqrt(m_qliAve[i]);
        });
}

float Steinhardt::normalizeSystem()
{
    float calc_norm(0);
    const auto normalizationfactor = float(4.0 * M_PI / m_num_ms);
    for (unsigned int k = 0; k < m_num_ms; ++k)
    {
        // Add the norm, which is the complex squared magnitude
        calc_norm += norm(m_qlm[k]);
    }
    const float ql_system_norm = std::sqrt(calc_norm * normalizationfactor);

    if (m_wl)
    {
        auto wigner3jvalues = getWigner3j(m_l);
        float wl_system_norm = reduceWigner3j(m_qlm.get(), m_l, wigner3jvalues);

        // The normalization factor of wl is calculated using qli, which is
        // equivalent to calculate the normalization factor from qlmi
        if (m_wl_normalize)
        {
            const float wl_normalization = std::sqrt(normalizationfactor) / ql_system_norm;
            wl_system_norm *= wl_normalization * wl_normalization * wl_normalization;
        }
        return wl_system_norm;
    }

    return ql_system_norm;
}

void Steinhardt::aggregatewl(util::ManagedArray<float>& target,
                             const util::ManagedArray<std::complex<float>>& source,
                             const util::ManagedArray<float>& normalization_source) const
{
    auto wigner3jvalues = getWigner3j(m_l);
    const auto normalizationfactor = float(4.0 * M_PI / m_num_ms);
    util::forLoopWrapper(0, m_Np, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
        {
            target[i] = reduceWigner3j(&(source({static_cast<unsigned int>(i), 0})), m_l, wigner3jvalues);
            if (m_wl_normalize)
            {
                const float normalization = std::sqrt(normalizationfactor) / normalization_source[i];
                target[i] *= normalization * normalization * normalization;
            }
        }
    });
}

}; }; // end namespace freud::order
