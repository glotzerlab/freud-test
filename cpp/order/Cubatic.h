// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef CUBATIC_H
#define CUBATIC_H

#include "ManagedArray.h"
#include "VectorMath.h"
#include <array>
#include <random>

/*! \file Cubatic.h
    \brief Compute the cubatic order parameter for each particle.
*/

namespace freud { namespace order {

//! Helper 4th-order tensor class for cubatic calculations.
/*! Strong orientational coordinates in the paper are defined as homogeneous
 *  4th order tensors constructed from tensor products of orbit vectors. The
 *  tensor4 class encapsulates some of the basic features required to enable
 *  these calculations, in particular the construction of the tensor from a
 *  vector and some arithmetic operations that help simplify the code.
 */
struct tensor4
{
    tensor4() = default;
    explicit tensor4(const vec3<float>& vector);
    tensor4 operator+=(const tensor4& b);
    tensor4 operator-(const tensor4& b) const;
    tensor4 operator*(const float& b) const;
    float& operator[](unsigned int index);

    void copyToManagedArray(util::ManagedArray<float>& ma);

    std::array<float, 81> data {0};
};

//! Compute the cubatic order parameter for a set of points
/*! The cubatic order parameter is defined according to the paper "Strong
 * orientational coordinates and orientational order parameters for symmetric
 * objects" by Amir Haji-Akbari and Sharon C. Glotzer
 * (http://dx.doi.org/10.1088/1751-8113/48/48/485201). Comments throughout this
 * file reference notes and equations from that paper for clarity. The central
 * idea is to define, for a given symmetry, a minimal set of vectors that can
 * be used to construct a coordinate system that has no degeneracies, i.e. one
 * that is equivalent for any two points that are identical up to any of the
 * transformations in the symmetry group. The set of vectors for a given rigid
 * body R is known as a symmetric descriptor of that object, and is constructed
 * from orbits of that vector (eq. 8). Strong orientational coordinates (SOCs)
 * are then constructed as homogeneous tensors constructed from this set (eq.
 * 3). The central idea of the paper is to then develop tensor functions of the
 * SOCs that can be used to quantify order.
 */
class Cubatic
{
public:
    //! Constructor
    Cubatic(float t_initial, float t_final, float scale, unsigned int replicates, unsigned int seed);

    //! Destructor
    ~Cubatic() = default;

    //! Reset the bond order array to all zeros
    void reset();

    //! Compute the cubatic order parameter
    void compute(quat<float>* orientations, unsigned int num_orientations);

    unsigned int getNumParticles() const
    {
        return m_n;
    }

    //! Get a reference to the last computed cubatic order parameter
    float getCubaticOrderParameter() const
    {
        return m_cubatic_order_parameter;
    }

    const util::ManagedArray<float>& getParticleOrderParameter() const
    {
        return m_particle_order_parameter;
    }

    const util::ManagedArray<float>& getGlobalTensor() const
    {
        return m_global_tensor;
    }

    const util::ManagedArray<float>& getCubaticTensor() const
    {
        return m_cubatic_tensor;
    }

    quat<float> getCubaticOrientation() const
    {
        return m_cubatic_orientation;
    }

    float getTInitial() const
    {
        return m_t_initial;
    }

    float getTFinal() const
    {
        return m_t_final;
    }

    float getScale() const
    {
        return m_scale;
    }

    unsigned int getNReplicates() const
    {
        return m_n_replicates;
    }

    unsigned int getSeed() const
    {
        return m_seed;
    }

private:
    //! Calculate the cubatic tensor
    /*! Implements the second line of eq. 27, the calculation of M_{\omega}.
     *  The cubatic tensor is computed by creating the homogeneous tensor
     *  corresponding to each basis vector rotated by the provided orientation
     *  and then summing all these resulting tensors.
     *
     *  \return The cubatic tensor M_{\omega}.
     */
    tensor4 calcCubaticTensor(quat<float>& orientation);

    //! Calculate the scalar cubatic order parameter.
    /*! Implements eq. 22.
     *
     *  \param cubatic_tensor The cubatic tensor M_{\omega}.
     *  \param global_tensor The tensor encoding the average system orientation (denoted \bar{M}).
     *
     *  \return The value of the cubatic order parameter.
     */
    static float calcCubaticOrderParameter(const tensor4& cubatic_tensor, const tensor4& global_tensor);

    //! Calculate the per-particle tensor.
    /*! Implements the first line of eq. 27, the calculation of M.
     *
     *  \param orientations The per-particle orientations.
     *
     *  \return The per-particle cubatic tensors value M^{ijkl}.
     */
    util::ManagedArray<tensor4> calculatePerParticleTensor(const quat<float>* orientations) const;

    //! Calculate the global tensor for the system.
    /*! Implements the third line of eq. 27, the calculation of \bar{M}.
     */
    tensor4 calculateGlobalTensor(quat<float>* orientations) const;

    //! Calculate a random quaternion.
    /*! To calculate a random quaternion in a way that obeys the right
     *  distribution of angles, we cannot simply just choose 4 random numbers
     *  and then normalize the quaternion. This function implements an
     *  appropriate calculation. It is templated to allow easy input of
     *  parameterized distributions using std::bind.
     */
    template<typename T> quat<float> calcRandomQuaternion(T& dist, float angle_multiplier = 1.0) const;

    float m_t_initial;           //!< Initial temperature for simulated annealing.
    float m_t_final;             //!< Final temperature for simulated annealing.
    float m_scale;               //!< Scaling factor to reduce temperature.
    unsigned int m_n_replicates; //!< Number of replicates.
    unsigned int m_seed;         //!< Random seed.
    unsigned int m_n {0};        //!< Last number of points computed.

    float m_cubatic_order_parameter {0}; //!< The value of the order parameter.
    quat<float> m_cubatic_orientation;   //!< The cubatic orientation.

    tensor4 m_gen_r4_tensor; //!< The sum of various products of Kronecker deltas that is stored as a member
                             //!< for convenient reuse.

    util::ManagedArray<float> m_particle_order_parameter; //!< The per-particle value of the order parameter.
    util::ManagedArray<float>
        m_global_tensor; //!< The system-averaged homogeneous tensor encoding all particle orientations.
    util::ManagedArray<float> m_cubatic_tensor; //!< The output tensor computed via simulated annealing.

    std::array<vec3<float>, 3>
        m_system_vectors; //!< The global coordinate system, always use a simple Euclidean basis.
};

}; }; // end namespace freud::order

#endif // CUBATIC_H
