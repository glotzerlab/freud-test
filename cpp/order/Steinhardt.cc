// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "Steinhardt.h"
#include "NeighborComputeFunctional.h"
#include "utils.h"
#include <vector>

/*! \file Steinhardt.cc
    \brief Computes variants of Steinhardt order parameters.
*/

namespace freud { namespace order {

// Calculating Ylm using fsph module
void Steinhardt::computeYlm(fsph::PointSPHEvaluator<float>& sph_eval, const float theta, const float phi,
                            std::vector<std::vector<std::complex<float>>>& Ylms) const
{
    sph_eval.compute(theta, phi);

    for (auto i = 0; i < m_ls.size(); ++i)
    {
        auto& Ylm = Ylms[i];
        auto l = m_ls[i];
        unsigned int m_index(0);

        for (typename fsph::PointSPHEvaluator<float>::iterator iter(sph_eval.begin_l(l, 0, true));
             iter != sph_eval.begin_l(l + 1, 0, true); ++iter)
        {
            // Manually add the Condon-Shortley phase, (-1)^m, to positive odd m
            float phase = 1;
            if (m_index <= l && m_index % 2 == 1)
            {
                phase = -1;
            }

            Ylm[m_index] = phase * (*iter);
            ++m_index;
        }
    }
}

void Steinhardt::reallocateArrays(unsigned int Np)
{
    m_Np = Np;

    for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        auto num_ms = m_num_ms[l_index];
        m_qlmi[l_index].prepare({Np, num_ms});
        m_qlm[l_index].prepare(num_ms);
        m_qli[l_index].prepare(Np);
        if (m_average)
        {
            m_qlmiAve[l_index].prepare({Np, num_ms});
            m_qliAve[l_index].prepare(Np);
        }
        if (m_wl)
        {
            m_wli[l_index].prepare(Np);
        }
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

    for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        m_qlm_local[l_index].reduceInto(m_qlm[l_index]);
    }

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
    std::vector<float> normalizationfactor(m_ls.size());
    for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        normalizationfactor[l_index] = float(4.0 * M_PI / m_num_ms[l_index]);
    }
    // For consistency, this reset is done here regardless of whether the array
    // is populated in baseCompute or computeAve.
    for (auto& qlm_local : m_qlm_local)
    {
        qlm_local.reset();
    }

    freud::locality::loopOverNeighborsIterator(
        points, points->getPoints(), m_Np, qargs, nlist,
        [=](size_t i, const std::shared_ptr<freud::locality::NeighborPerPointIterator>& ppiter) {
            float total_weight(0);
            const vec3<float> ref((*points)[i]);
            // Construct PointSPHEvaluator outside loop since the construction is costly.
            auto max_l = std::max_element(m_ls.begin(), m_ls.end());
            if (max_l == m_ls.end())
            {
                throw std::invalid_argument("Selected l's do not have a maximum.");
            }
            fsph::PointSPHEvaluator<float> sph_eval(*max_l);

            std::vector<std::vector<std::complex<float>>> Ylms(m_ls.size());
            for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
            {
                Ylms[l_index].resize(m_num_ms[l_index]);
            }

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

                computeYlm(sph_eval, theta, phi, Ylms); // Fill up Ylm

                for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
                {
                    auto& qlmi = m_qlmi[l_index];
                    auto& Ylm = Ylms[l_index];
                    // Get the initial index and iterate using ++ for faster iteration
                    // Profling showed using operator() to slow the code significantly.
                    auto index = qlmi.getIndex({static_cast<unsigned int>(i), 0});
                    for (unsigned int k = 0; k < m_num_ms[l_index]; ++k)
                    {
                        qlmi[index] += weight * Ylm[k];
                        ++index;
                    }
                }
                // Accumulate weight for normalization
                total_weight += weight;
            } // End loop going over neighbor bonds

            // Normalize!
            for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
            {
                // get l specific vectors/arrays
                auto& qlmi = m_qlmi[l_index];
                auto& qli = m_qli[l_index];
                auto& qlm_local = m_qlm_local[l_index];
                unsigned int index = qlmi.getIndex({static_cast<unsigned int>(i), 0});

                for (unsigned int k = 0; k < m_num_ms[l_index]; ++k)
                {
                    // Cache the index for efficiency.
                    qlmi[index] /= total_weight;
                    // Add the norm, which is the (complex) squared magnitude
                    qli[i] += norm(qlmi[index]);
                    // This array gets populated by computeAve in the averaging case.
                    if (!m_average)
                    {
                        qlm_local.local()[k] += qlmi[index] / float(m_Np);
                    }
                    ++index;
                }
                qli[i] *= normalizationfactor[l_index];
                qli[i] = std::sqrt(qli[i]);
            }
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

    std::vector<float> normalizationfactor(m_ls.size());
    for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        normalizationfactor[l_index] = float(4.0 * M_PI / m_num_ms[l_index]);
    }

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
                    for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
                    {
                        auto& qlmiAve = m_qlmiAve[l_index];
                        auto& qlmi = m_qlmi[l_index];
                        for (unsigned int k = 0; k < m_num_ms[l_index]; ++k)
                        {
                            // Adding all the qlm of the neighbors. We use the
                            // vector function signature for indexing into the
                            // arrays for speed.
                            qlmiAve({static_cast<unsigned int>(i), k}) += qlmi({nb2.point_idx, k});
                        }
                    }
                    neighborcount++;
                } // End loop over particle neighbor's bonds
            }     // End loop over particle's bonds

            // Normalize!

            for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
            {
                auto& qlmiAve = m_qlmiAve[l_index];
                auto& qlmi = m_qlmi[l_index];
                auto& qlm_local = m_qlm_local[l_index];
                auto& qliAve = m_qliAve[l_index];

                for (unsigned int k = 0; k < m_num_ms[l_index]; ++k)
                {
                    // Cache the index for efficiency.
                    const unsigned int index = qlmiAve.getIndex({static_cast<unsigned int>(i), k});
                    // Adding the qlm of the particle i itself
                    qlmiAve[index] += qlmi[index];
                    qlmiAve[index] /= static_cast<float>(neighborcount);
                    qlm_local.local()[k] += qlmiAve[index] / float(m_Np);
                    // Add the norm, which is the complex squared magnitude
                    qliAve[i] += norm(qlmiAve[index]);
                }
                qliAve[i] *= normalizationfactor[l_index];
                qliAve[i] = std::sqrt(qliAve[i]);
            }
        });
}

std::vector<float> Steinhardt::normalizeSystem()
{
    std::vector<float> system_norms;
    for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        auto& qlm = m_qlm[l_index];
        auto l = m_ls[l_index];
        float calc_norm(0);
        const auto normalizationfactor = float(4.0 * M_PI / m_num_ms[l_index]);
        for (unsigned int k = 0; k < m_num_ms[l_index]; ++k)
        {
            // Add the norm, which is the complex squared magnitude
            calc_norm += norm(qlm[k]);
        }
        const float ql_system_norm = std::sqrt(calc_norm * normalizationfactor);

        if (m_wl)
        {
            auto wigner3jvalues = getWigner3j(l);
            float wl_system_norm = reduceWigner3j(qlm.get(), l, wigner3jvalues);

            // The normalization factor of wl is calculated using qli, which is
            // equivalent to calculate the normalization factor from qlmi
            if (m_wl_normalize)
            {
                const float wl_normalization = std::sqrt(normalizationfactor) / ql_system_norm;
                wl_system_norm *= wl_normalization * wl_normalization * wl_normalization;
            }
            system_norms.push_back(wl_system_norm);
        }
        else
        {
            system_norms.push_back(ql_system_norm);
        }
    }
    return system_norms;
}

void Steinhardt::aggregatewl(std::vector<util::ManagedArray<float>>& target,
                             const std::vector<util::ManagedArray<std::complex<float>>>& source,
                             const std::vector<util::ManagedArray<float>>& normalization_source) const
{
    for (auto l_index = 0; l_index < m_ls.size(); ++l_index)
    {
        auto l = m_ls[l_index];
        const auto& source_l = source[l_index];
        auto& target_l = target[l_index];
        const auto& normalization_l = normalization_source[l_index];

        const auto normalizationfactor = float(4.0 * M_PI / m_num_ms[l_index]);
        auto wigner3jvalues = getWigner3j(l);

        util::forLoopWrapper(0, m_Np, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                target_l[i]
                    = reduceWigner3j(&(source_l({static_cast<unsigned int>(i), 0})), l, wigner3jvalues);
                if (m_wl_normalize)
                {
                    const float normalization = std::sqrt(normalizationfactor) / normalization_l[i];
                    target_l[i] *= normalization * normalization * normalization;
                }
            }
        });
    }
}

}; }; // end namespace freud::order
