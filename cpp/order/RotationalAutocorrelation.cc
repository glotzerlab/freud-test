// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include "RotationalAutocorrelation.h"

#include "utils.h"
#include <cmath>

/*! \file RotationalAutocorrelation.cc
    \brief Implements the RotationalAutocorrelation class.
*/

namespace freud { namespace order {

// This function wraps exponentiation for complex numbers to avoid
// the case where pow((0,0), 0) returns (nan, nan). Additionally, it
// serves as a micro-optimization since std::pow can be slow. We can take the
// exponent to be an unsigned int since all the sums in the hyperspherical
// harmonic below guarantee this.
inline std::complex<float> cpow(std::complex<float> base, unsigned int p)
{
    if (p == 0)
    {
        return {1, 0};
    }
    std::complex<float> val(base);
    for (unsigned int i = 1; i < p; i++)
    {
        val *= base;
    }
    return val;
}

inline std::complex<float> RotationalAutocorrelation::hypersphere_harmonic(const std::complex<float> xi,
                                                                           std::complex<float> zeta,
                                                                           const unsigned int m1,
                                                                           const unsigned int m2)
{
    const std::complex<float> xi_conj = std::conj(xi);
    const std::complex<float> zeta_conj = std::conj(zeta);

    // Doing a summation over non-negative exponents, which requires the additional inner conditional.
    std::complex<float> sum_tracker(0, 0);
    for (unsigned int k = (m1 + m2 < m_l ? 0 : m1 + m2 - m_l); k <= std::min(m1, m2); k++)
    {
        float fact_product = static_cast<float>(m_factorials[k]) * m_factorials[m_l + k - m1 - m2]
            * m_factorials[m1 - k] * m_factorials[m2 - k];
        sum_tracker += cpow(xi_conj, k) * cpow(zeta, m2 - k) * cpow(zeta_conj, m1 - k)
            * cpow(-xi, m_l + k - m1 - m2) / fact_product;
    }
    return sum_tracker;
}

void RotationalAutocorrelation::compute(const quat<float>* ref_orientations, const quat<float>* orientations,
                                        unsigned int N)
{
    m_RA_array.prepare(N);

    // Precompute the hyperspherical harmonics for the unit quaternion. The
    // default quaternion constructor gives a unit quaternion. We will assume
    // the same iteration order here as in the loop below to save ourselves
    // from having to use a more expensive process (i.e. a map).
    std::complex<float> xi = std::complex<float>(0, 0);
    std::complex<float> zeta = std::complex<float>(0, 1);
    std::vector<std::complex<float>> unit_harmonics;
    std::vector<std::vector<float>> prefactors(m_l + 1, std::vector<float>(m_l + 1, float(0)));
    for (unsigned int a = 0; a <= m_l; a++)
    {
        for (unsigned int b = 0; b <= m_l; b++)
        {
            unit_harmonics.push_back(std::conj(hypersphere_harmonic(xi, zeta, a, b)));
            prefactors[a][b] = static_cast<float>(m_factorials[a] * m_factorials[m_l - a] * m_factorials[b]
                                                  * m_factorials[m_l - b])
                / (static_cast<float>(m_l + 1));
        }
    }

    // Parallel loop is over orientations (technically (ref_or, or) pairs).
    util::forLoopWrapper(0, N, [=](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
        {
            // Transform the orientation quaternions into Xi/Zeta coordinates;
            quat<float> qq_1 = conj(ref_orientations[i]) * orientations[i];
            std::complex<float> xi = std::complex<float>(qq_1.v.x, qq_1.v.y);
            std::complex<float> zeta = std::complex<float>(qq_1.v.z, qq_1.s);

            // Loop through the valid quantum numbers.
            m_RA_array[i] = std::complex<float>(0, 0);
            unsigned int uh_index = 0;
            for (unsigned int a = 0; a <= m_l; a++)
            {
                for (unsigned int b = 0; b <= m_l; b++)
                {
                    std::complex<float> combined_value
                        = unit_harmonics[uh_index] * hypersphere_harmonic(xi, zeta, a, b);
                    m_RA_array[i] += prefactors[a][b] * combined_value;
                    uh_index += 1;
                }
            }
        }
    });

    float RA_sum(0);
    for (unsigned int i = 0; i < N; i++)
    {
        RA_sum += std::real(m_RA_array[i]);
    }
    m_Ft = RA_sum / static_cast<float>(N);
};

}; }; // end namespace freud::order
