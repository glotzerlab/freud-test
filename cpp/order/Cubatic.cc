// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#include <cstring>
#include <functional>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "Cubatic.h"
#include "utils.h"

/*! \file Cubatic.h
    \brief Compute the cubatic order parameter for each particle.
*/

namespace freud { namespace order {

tensor4::tensor4(const vec3<float>& vector)
{
    unsigned int cnt = 0;
    std::array<float, 3> v = {vector.x, vector.y, vector.z};
    for (float vi : v)
    {
        for (float vj : v)
        {
            for (float vk : v)
            {
                for (float vl : v)
                {
                    data[cnt] = vi * vj * vk * vl;
                    cnt++;
                }
            }
        }
    }
}

//! Writeable index into array.
float& tensor4::operator[](unsigned int index)
{
    return data[index];
}

tensor4 tensor4::operator+=(const tensor4& b)
{
    for (unsigned int i = 0; i < 81; i++)
    {
        data[i] += b.data[i];
    }
    return *this;
}

tensor4 tensor4::operator-(const tensor4& b) const
{
    tensor4 c;
    for (unsigned int i = 0; i < 81; i++)
    {
        c.data[i] = data[i] - b.data[i];
    }
    return c;
}

tensor4 tensor4::operator*(const float& b) const
{
    tensor4 c;
    for (unsigned int i = 0; i < 81; i++)
    {
        c.data[i] = data[i] * b;
    }
    return c;
}

void tensor4::copyToManagedArray(util::ManagedArray<float>& ma)
{
    std::copy(data.begin(), data.end(), ma.get());
}

//! Complete tensor contraction.
/*! This function is simply a sum-product over two tensors. For reference, see eq. 4.
 *
 *  \param a The first tensor.
 *  \param b The second tensor.
 */
float dot(const tensor4& a, const tensor4& b)
{
    float c = 0;
    for (unsigned int i = 0; i < 81; i++)
    {
        c += a.data[i] * b.data[i];
    }
    return c;
}

//! Generate the r4 tensor.
/*! The r4 tensor is not a word used in the paper, but is a name introduced in
 *  this code to refer to the second term in eqs. 27 in the paper. It is simply
 *  a scaled sum of some delta function products. For convenience, its
 *  calculation is performed in a single function.
 */
tensor4 genR4Tensor()
{
    // Construct the identity matrix to build the delta functions.
    util::ManagedArray<float> identity({3, 3});
    identity(0, 0) = 1;
    identity(1, 1) = 1;
    identity(2, 2) = 1;

    unsigned int cnt = 0;
    tensor4 r4 = tensor4();
    for (unsigned int i = 0; i < 3; ++i)
    {
        for (unsigned int j = 0; j < 3; ++j)
        {
            for (unsigned int k = 0; k < 3; ++k)
            {
                for (unsigned int l = 0; l < 3; ++l)
                {
                    // ijkl term
                    r4[cnt] += identity(i, j) * identity(k, l);
                    // ikjl term
                    r4[cnt] += identity(i, k) * identity(j, l);
                    // iljk term
                    r4[cnt] += identity(i, l) * identity(j, k);
                    r4[cnt] *= 2.0 / 5.0;
                    ++cnt;
                }
            }
        }
    }
    return r4;
}

Cubatic::Cubatic(float t_initial, float t_final, float scale, unsigned int n_replicates, unsigned int seed)
    : m_t_initial(t_initial), m_t_final(t_final), m_scale(scale), m_n_replicates(n_replicates), m_seed(seed)
{
    if (m_t_initial < m_t_final)
    {
        throw std::invalid_argument("Cubatic requires that t_initial must be greater than t_final.");
    }
    if (t_final < 1e-6)
    {
        throw std::invalid_argument("Cubatic requires that t_final must be >= 1e-6.");
    }
    // cppcheck erroneously flags this check as redundant because it follows
    // both branches independently.
    // cppcheck-suppress knownConditionTrueFalse
    if ((scale >= 1) || (scale <= 0))
    {
        throw std::invalid_argument("Cubatic requires that scale must be between 0 and 1.");
    }

    m_gen_r4_tensor = genR4Tensor();

    // Initialize the system vectors using Euclidean vectors.
    m_system_vectors[0] = vec3<float>(1, 0, 0);
    m_system_vectors[1] = vec3<float>(0, 1, 0);
    m_system_vectors[2] = vec3<float>(0, 0, 1);
}

tensor4 Cubatic::calcCubaticTensor(quat<float>& orientation)
{
    tensor4 calculated_tensor = tensor4();
    for (auto& m_system_vector : m_system_vectors)
    {
        calculated_tensor += tensor4(rotate(orientation, m_system_vector));
    }
    return calculated_tensor * float(2.0) - m_gen_r4_tensor;
}

float Cubatic::calcCubaticOrderParameter(const tensor4& cubatic_tensor, const tensor4& global_tensor)
{
    tensor4 diff = global_tensor - cubatic_tensor;
    return float(1.0) - dot(diff, diff) / dot(cubatic_tensor, cubatic_tensor);
}

template<typename T> quat<float> Cubatic::calcRandomQuaternion(T& dist, float angle_multiplier) const
{
    float theta = 2.0 * M_PI * dist();
    float phi = std::acos(2.0 * dist() - 1.0);
    vec3<float> axis
        = vec3<float>(std::cos(theta) * std::sin(phi), std::sin(theta) * std::sin(phi), std::cos(phi));
    float axis_norm = std::sqrt(dot(axis, axis));
    axis /= axis_norm;
    float angle = angle_multiplier * dist();
    return quat<float>::fromAxisAngle(axis, angle);
}

util::ManagedArray<tensor4> Cubatic::calculatePerParticleTensor(const quat<float>* orientations) const
{
    util::ManagedArray<tensor4> particle_tensor(m_n);

    // calculate per-particle tensor
    util::forLoopWrapper(0, m_n, [=, &particle_tensor](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
        {
            tensor4 l_mbar = tensor4();
            for (const auto& m_system_vector : m_system_vectors)
            {
                // Calculate the homogeneous tensor H for each vector then add
                // to the per-particle value.
                vec3<float> v_r = rotate(orientations[i], m_system_vector);
                tensor4 r4_tensor(v_r);
                l_mbar += r4_tensor;
            }

            // Apply the prefactor from the sum in equation 27 before assigning.
            particle_tensor[i] = l_mbar * 2.0;
        }
    });
    return particle_tensor;
}

tensor4 Cubatic::calculateGlobalTensor(quat<float>* orientations) const
{
    tensor4 global_tensor = tensor4();
    util::ManagedArray<tensor4> particle_tensor = calculatePerParticleTensor(orientations);

    // now calculate the global tensor
    float n_inv = float(1.0) / static_cast<float>(m_n);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, 81),
                      [=, &global_tensor, &n_inv, &particle_tensor](const tbb::blocked_range<size_t>& r) {
                          for (size_t i = r.begin(); i != r.end(); i++)
                          {
                              float tensor_value = 0;
                              for (unsigned int j = 0; j < m_n; j++)
                              {
                                  tensor_value += particle_tensor[j][i];
                              }
                              // Note that in the third equation in eq. 27, the prefactor of the
                              // sum is 2/N, but the factor of 2 is already accounted for in the
                              // calculation of per particle calculation in
                              // calculatePerParticleTensor, so here we just need to apply the
                              // 1/N scaling.
                              global_tensor[i] = tensor_value * n_inv;
                          }
                      });
    return global_tensor - m_gen_r4_tensor;
}

void Cubatic::compute(quat<float>* orientations, unsigned int num_orientations)
{
    m_n = num_orientations;
    m_particle_order_parameter.prepare(m_n);

    // Calculate the per-particle tensor
    tensor4 global_tensor = calculateGlobalTensor(orientations);
    m_global_tensor.prepare({3, 3, 3, 3});
    global_tensor.copyToManagedArray(m_global_tensor);

    // The paper recommends using a Newton-Raphson scheme to optimize the order
    // parameter, but in practice we find that simulated annealing performs
    // much better, so we perform replicates of the process and choose the best
    // one.
    util::ManagedArray<tensor4> p_cubatic_tensor(m_n_replicates);
    util::ManagedArray<float> p_cubatic_order_parameter(m_n_replicates);
    util::ManagedArray<quat<float>> p_cubatic_orientation(m_n_replicates);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, m_n_replicates),
        [=, &p_cubatic_orientation, &p_cubatic_order_parameter,
         &p_cubatic_tensor](const tbb::blocked_range<size_t>& r) {
            // create thread-specific rng
            unsigned int thread_start = r.begin();

            std::vector<unsigned int> seed_seq(3);
            seed_seq[0] = m_seed;
            seed_seq[1] = thread_start;
            seed_seq[2] = 0xffaabb;
            std::seed_seq seed(seed_seq.begin(), seed_seq.end());
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> base_dist(0, 1);
            auto dist = [&]() { return base_dist(rng); };

            for (size_t i = r.begin(); i != r.end(); i++)
            {
                // need to generate random orientation
                quat<float> cubatic_orientation = calcRandomQuaternion(dist);
                quat<float> new_orientation = cubatic_orientation;

                // now calculate the cubatic tensor
                tensor4 cubatic_tensor = calcCubaticTensor(cubatic_orientation);
                float cubatic_order_parameter = calcCubaticOrderParameter(cubatic_tensor, global_tensor);
                float new_order_parameter = cubatic_order_parameter;

                // set initial temperature and count
                float t_current = m_t_initial;
                unsigned int loop_count = 0;
                // simulated annealing loop; loop counter to prevent inf loops
                while ((t_current > m_t_final) && (loop_count < 10000))
                {
                    ++loop_count;
                    new_orientation = calcRandomQuaternion(dist, 0.1) * (cubatic_orientation);
                    // now calculate the cubatic tensor
                    tensor4 new_cubatic_tensor = calcCubaticTensor(new_orientation);
                    new_order_parameter = calcCubaticOrderParameter(new_cubatic_tensor, global_tensor);
                    if (new_order_parameter > cubatic_order_parameter)
                    {
                        cubatic_tensor = new_cubatic_tensor;
                        cubatic_order_parameter = new_order_parameter;
                        cubatic_orientation = new_orientation;
                    }
                    else
                    {
                        float boltzmann_factor
                            = std::exp(-(cubatic_order_parameter - new_order_parameter) / t_current);
                        if (boltzmann_factor >= dist())
                        {
                            cubatic_tensor = new_cubatic_tensor;
                            cubatic_order_parameter = new_order_parameter;
                            cubatic_orientation = new_orientation;
                        }
                        else
                        {
                            continue;
                        }
                    }
                    t_current *= m_scale;
                }
                // set values
                p_cubatic_tensor[i] = cubatic_tensor;
                p_cubatic_orientation[i].s = cubatic_orientation.s;
                p_cubatic_orientation[i].v = cubatic_orientation.v;
                p_cubatic_order_parameter[i] = cubatic_order_parameter;
            }
        });

    // Loop over threads and choose the replicate that found the highest order.
    unsigned int max_idx = 0;
    float max_cubatic_order_parameter = p_cubatic_order_parameter[max_idx];
    for (unsigned int i = 1; i < m_n_replicates; ++i)
    {
        if (p_cubatic_order_parameter[i] > max_cubatic_order_parameter)
        {
            max_idx = i;
            max_cubatic_order_parameter = p_cubatic_order_parameter[i];
        }
    }

    m_cubatic_tensor.prepare({3, 3, 3, 3});
    p_cubatic_tensor[max_idx].copyToManagedArray(m_cubatic_tensor);
    m_cubatic_orientation = p_cubatic_orientation[max_idx];
    m_cubatic_order_parameter = p_cubatic_order_parameter[max_idx];

    // Now calculate the per-particle order parameters
    tbb::parallel_for(tbb::blocked_range<size_t>(0, m_n), [=](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); i++)
        {
            // The per-particle order parameter is defined as the value of the
            // cubatic order parameter if the global orientation was the
            // particle orientation, so we can reuse the same machinery.
            m_particle_order_parameter[i]
                = calcCubaticOrderParameter(calcCubaticTensor(orientations[i]), global_tensor);
        }
    });
}

}; }; // end namespace freud::order
