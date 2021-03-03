// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "HexaticTranslational.h"

namespace freud { namespace order {

//! Compute the order parameter
template<typename T>
template<typename Func>
void HexaticTranslational<T>::computeGeneral(Func func, const freud::locality::NeighborList* nlist,
                                             const freud::locality::NeighborQuery* points,
                                             freud::locality::QueryArgs qargs, bool normalize_by_k)
{
    const auto box = points->getBox();
    box.enforce2D();

    const unsigned int Np = points->getNPoints();

    m_psi_array.prepare(Np);

    freud::locality::loopOverNeighborsIterator(
        points, points->getPoints(), Np, qargs, nlist,
        [=](size_t i, const std::shared_ptr<freud::locality::NeighborPerPointIterator>& ppiter) {
            float total_weight(0);
            const vec3<float> ref((*points)[i]);

            for (freud::locality::NeighborBond nb = ppiter->next(); !ppiter->end(); nb = ppiter->next())
            {
                // Compute vector from query_point to point
                const vec3<float> delta = box.wrap((*points)[nb.point_idx] - ref);
                const float weight(m_weighted ? nb.weight : 1.0);

                // Compute psi for this vector
                m_psi_array[i] += weight * func(delta);
                total_weight += weight;
            }
            if (normalize_by_k)
            {
                m_psi_array[i] /= std::complex<float>(m_k);
            }
            else
            {
                m_psi_array[i] /= std::complex<float>(total_weight);
            }
        });
}

Hexatic::Hexatic(unsigned int k, bool weighted) : HexaticTranslational<unsigned int>(k, weighted) {}

void Hexatic::compute(const freud::locality::NeighborList* nlist,
                      const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    computeGeneral(
        [this](const vec3<float>& delta) {
            const float theta_ij = std::atan2(delta.y, delta.x);
            return std::exp(std::complex<float>(0, m_k * theta_ij));
        },
        nlist, points, qargs, false);
}

Translational::Translational(float k, bool weighted) : HexaticTranslational<float>(k, weighted) {}

void Translational::compute(const freud::locality::NeighborList* nlist,
                            const freud::locality::NeighborQuery* points, freud::locality::QueryArgs qargs)
{
    computeGeneral([](const vec3<float>& delta) { return std::complex<float>(delta.x, delta.y); }, nlist,
                   points, qargs, true);
}

}; }; // namespace freud::order
