// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef BOX_H
#define BOX_H

#include "utils.h"
#include <algorithm>
#include <complex>
#include <sstream>
#include <stdexcept>

#include "VectorMath.h"

/*! \file Box.h
    \brief Represents simulation boxes and contains helpful wrapping functions.
*/

namespace freud { namespace constants {
// Constant 2*pi for convenient use everywhere.
constexpr float TWO_PI = 2.0 * M_PI;
}; }; // end namespace freud::constants

namespace freud { namespace box {

//! Stores box dimensions and provides common routines for wrapping vectors back into the box
/*! Box stores a standard HOOMD simulation box that goes from -L/2 to L/2 in each dimension, allowing Lx, Ly,
 Lz, and triclinic tilt factors xy, xz, and yz to be specified independently.
 *

    A number of utility functions are provided to work with coordinates in boxes. These are provided as
 inlined methods in the header file so they can be called in inner loops without sacrificing performance.
     - wrap()
     - unwrap()

    A Box can represent either a two or three dimensional box. By default, a Box is 3D, but can be set as 2D
 with the method set2D(), or via an optional boolean argument to the constructor. is2D() queries if a Box is
 2D or not. 2D boxes have a "volume" of Lx * Ly, and Lz is set to 0. To keep programming simple, all inputs
 and outputs are still 3-component vectors even for 2D boxes. The third component ignored (assumed set to 0).
*/
class Box
{
public:
    //! Nullary constructor for Cython
    Box()
    {
        m_2d = false; // Assign before calling setL!
        setL(0, 0, 0);
        m_periodic = vec3<bool>(true, true, true);
        m_xy = m_xz = m_yz = 0;
    }

    //! Construct a square/cubic box
    explicit Box(float L, bool _2d = false)
    {
        m_2d = _2d; // Assign before calling setL!
        setL(L, L, L);
        m_periodic = vec3<bool>(true, true, true);
        m_xy = m_xz = m_yz = 0;
    }

    //! Construct an orthorhombic box
    Box(float Lx, float Ly, float Lz, bool _2d = false)
    {
        m_2d = _2d; // Assign before calling setL!
        setL(Lx, Ly, Lz);
        m_periodic = vec3<bool>(true, true, true);
        m_xy = m_xz = m_yz = 0;
    }

    //! Construct a triclinic box
    Box(float Lx, float Ly, float Lz, float xy, float xz, float yz, bool _2d = false)
    {
        m_2d = _2d; // Assign before calling setL!
        setL(Lx, Ly, Lz);
        m_periodic = vec3<bool>(true, true, true);
        m_xy = xy;
        m_xz = xz;
        m_yz = yz;
    }

    inline bool operator==(const Box& b) const
    {
        return ((this->getL() == b.getL()) && (this->getTiltFactorXY() == b.getTiltFactorXY())
                && (this->getTiltFactorXZ() == b.getTiltFactorXZ())
                && (this->getTiltFactorYZ() == b.getTiltFactorYZ()));
    }

    inline bool operator!=(const Box& b) const
    {
        return ((this->getL() != b.getL()) || (this->getTiltFactorXY() != b.getTiltFactorXY())
                || (this->getTiltFactorXZ() != b.getTiltFactorXZ())
                || (this->getTiltFactorYZ() != b.getTiltFactorYZ()));
    }

    //! Set L, box lengths, inverses.  Box is also centered at zero.
    void setL(const vec3<float>& L)
    {
        setL(L.x, L.y, L.z);
    }

    //! Set L, box lengths, inverses.  Box is also centered at zero.
    void setL(const float Lx, const float Ly, const float Lz)
    {
        if (m_2d)
        {
            m_L = vec3<float>(Lx, Ly, 0);
            m_Linv = vec3<float>(1 / m_L.x, 1 / m_L.y, 0);
        }
        else
        {
            m_L = vec3<float>(Lx, Ly, Lz);
            m_Linv = vec3<float>(1 / m_L.x, 1 / m_L.y, 1 / m_L.z);
        }

        m_hi = m_L / float(2.0);
        m_lo = -m_hi;
    }

    //! Set whether box is 2D
    void set2D(bool _2d)
    {
        m_2d = _2d;
        m_L.z = 0;
        m_Linv.z = 0;
    }

    //! Returns whether box is two dimensional
    bool is2D() const
    {
        return m_2d;
    }

    //! Get the value of Lx
    float getLx() const
    {
        return m_L.x;
    }

    //! Get the value of Ly
    float getLy() const
    {
        return m_L.y;
    }

    //! Get the value of Lz
    float getLz() const
    {
        return m_L.z;
    }

    //! Get current L
    vec3<float> getL() const
    {
        return m_L;
    }

    //! Get current stored inverse of L
    vec3<float> getLinv() const
    {
        return m_Linv;
    }

    //! Get tilt factor xy
    float getTiltFactorXY() const
    {
        return m_xy;
    }

    //! Get tilt factor xz
    float getTiltFactorXZ() const
    {
        return m_xz;
    }

    //! Get tilt factor yz
    float getTiltFactorYZ() const
    {
        return m_yz;
    }

    //! Set tilt factor xy
    void setTiltFactorXY(float xy)
    {
        m_xy = xy;
    }

    //! Set tilt factor xz
    void setTiltFactorXZ(float xz)
    {
        m_xz = xz;
    }

    //! Set tilt factor yz
    void setTiltFactorYZ(float yz)
    {
        m_yz = yz;
    }

    //! Get the volume of the box (area in 2D)
    float getVolume() const
    {
        if (m_2d)
        {
            return m_L.x * m_L.y;
        }
        return m_L.x * m_L.y * m_L.z;
    }

    //! Convert fractional coordinates into absolute coordinates
    /*! \param f Fractional coordinates between 0 and 1 within
     *         parallelepipedal box
     *  \return A vector inside the box corresponding to f
     */
    vec3<float> makeAbsolute(const vec3<float>& f) const
    {
        vec3<float> v = m_lo + f * m_L;
        v.x += m_xy * v.y + m_xz * v.z;
        v.y += m_yz * v.z;
        if (m_2d)
        {
            v.z = float(0.0);
        }
        return v;
    }

    //! Convert fractional coordinates into absolute coordinates in place
    /*! \param vecs Vectors of fractional coordinates between 0 and 1 within
     *         parallelepipedal box
     *  \param Nvecs Number of vectors
     */
    void makeAbsolute(vec3<float>* vecs, unsigned int Nvecs) const
    {
        util::forLoopWrapper(0, Nvecs, [=](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                vecs[i] = makeAbsolute(vecs[i]);
            }
        });
    }

    //! Compute the position of the particle in box relative coordinates
    /*! \param p point
     *  \returns alpha
     *
     *  alpha.x is 0 when \a x is on the far left side of the box and
     *  1.0 when it is on the far right. If x is outside of the box in
     *  either direction, it will go larger than 1 or less than 0
     *  keeping the same scaling.
     */
    vec3<float> makeFractional(const vec3<float>& v,
                               const vec3<float>& ghost_width = vec3<float>(0.0, 0.0, 0.0)) const
    {
        vec3<float> delta = v - m_lo;
        delta.x -= (m_xz - m_yz * m_xy) * v.z + m_xy * v.y;
        delta.y -= m_yz * v.z;
        delta = (delta + ghost_width) / (m_L + float(2.0) * ghost_width);

        if (m_2d)
        {
            delta.z = float(0.0);
        }
        return delta;
    }

    void makeFractional(vec3<float>* vecs, unsigned int Nvecs) const
    {
        util::forLoopWrapper(0, Nvecs, [=](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                vecs[i] = makeFractional(vecs[i]);
            }
        });
    }

    //! Get periodic image of a vector.
    /*! \param v The vector to check.
     *  \param image The image of a given point.
     */
    inline void getImage(const vec3<float>& v, vec3<int>& image) const
    {
        vec3<float> f = makeFractional(v) - vec3<float>(0.5, 0.5, 0.5);
        if (m_2d)
        {
            f.z = float(0.0);
        }
        image.x = (int) ((f.x >= float(0.0)) ? f.x + float(0.5) : f.x - float(0.5));
        image.y = (int) ((f.y >= float(0.0)) ? f.y + float(0.5) : f.y - float(0.5));
        image.z = (int) ((f.z >= float(0.0)) ? f.z + float(0.5) : f.z - float(0.5));
    }

    //! Get the periodic image vectors belongs to
    /*! \param vecs The vectors to check
     *  \param Nvecs Number of vectors
        \param res Array to save the images
     */
    void getImages(vec3<float>* vecs, unsigned int Nvecs, vec3<int>* res) const
    {
        util::forLoopWrapper(0, Nvecs, [=](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                getImage(vecs[i], res[i]);
            }
        });
    }

    //! Wrap a vector back into the box
    /*! \param v Vector to wrap, updated to the minimum image obeying the periodic settings
     *  \returns Wrapped vector
     */
    vec3<float> wrap(const vec3<float>& v) const
    {
        // Return quickly if the box is aperiodic
        if (!m_periodic.x && !m_periodic.y && !m_periodic.z)
        {
            return v;
        }

        vec3<float> v_frac = makeFractional(v);
        if (m_periodic.x)
        {
            v_frac.x = util::modulusPositive(v_frac.x, float(1.0));
        }
        if (m_periodic.y)
        {
            v_frac.y = util::modulusPositive(v_frac.y, float(1.0));
        }
        if (m_periodic.z)
        {
            v_frac.z = util::modulusPositive(v_frac.z, float(1.0));
        }
        return makeAbsolute(v_frac);
    }

    //! Wrap vectors back into the box in place
    /*! \param vecs Vectors to wrap, updated to the minimum image obeying the periodic settings
     *  \param Nvecs Number of vectors
     */
    void wrap(vec3<float>* vecs, unsigned int Nvecs) const
    {
        util::forLoopWrapper(0, Nvecs, [=](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                vecs[i] = wrap(vecs[i]);
            }
        });
    }

    //! Unwrap given positions to their absolute location in place
    /*! \param vecs Vectors of coordinates to unwrap
     *  \param images images flags for this point
        \param Nvecs Number of vectors
    */
    void unwrap(vec3<float>* vecs, const vec3<int>* images, unsigned int Nvecs) const
    {
        util::forLoopWrapper(0, Nvecs, [=](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                vecs[i] += getLatticeVector(0) * float(images[i].x);
                vecs[i] += getLatticeVector(1) * float(images[i].y);
                if (!m_2d)
                {
                    vecs[i] += getLatticeVector(2) * float(images[i].z);
                }
            }
        });
    }

    //! Compute center of mass for vectors
    /*! \param vecs Vectors to compute center of mass
     *  \param Nvecs Number of vectors
     *  \param masses Optional array of masses, of length Nvecs
     *  \return Center of mass as a vec3<float>
     */
    vec3<float> centerOfMass(vec3<float>* vecs, size_t Nvecs, const float* masses = nullptr) const
    {
        // This roughly follows the implementation in
        // https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions
        float total_mass(0);
        vec3<std::complex<float>> xi_mean;

        for (size_t i = 0; i < Nvecs; ++i)
        {
            vec3<float> phase(constants::TWO_PI * makeFractional(vecs[i]));
            vec3<std::complex<float>> xi(std::polar(float(1.0), phase.x), std::polar(float(1.0), phase.y),
                                         std::polar(float(1.0), phase.z));
            float mass = (masses != nullptr) ? masses[i] : float(1.0);
            total_mass += mass;
            xi_mean += std::complex<float>(mass, 0) * xi;
        }
        xi_mean /= std::complex<float>(total_mass, 0);

        return wrap(makeAbsolute(vec3<float>(std::arg(xi_mean.x), std::arg(xi_mean.y), std::arg(xi_mean.z))
                                 / constants::TWO_PI));
    }

    //! Subtract center of mass from vectors
    /*! \param vecs Vectors to center
     *  \param Nvecs Number of vectors
     *  \param masses Optional array of masses, of length Nvecs
     */
    void center(vec3<float>* vecs, unsigned int Nvecs, const float* masses = nullptr) const
    {
        vec3<float> com(centerOfMass(vecs, Nvecs, masses));
        util::forLoopWrapper(0, Nvecs, [=](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                vecs[i] = wrap(vecs[i] - com);
            }
        });
    }

    //! Calculate distance between two points using boundary conditions
    /*! \param r_i Position of first point
        \param r_j Position of second point
    */
    inline float computeDistance(const vec3<float>& r_i, const vec3<float>& r_j) const
    {
        const vec3<float> r_ij = wrap(r_j - r_i);
        return std::sqrt(dot(r_ij, r_ij));
    }

    //! Calculate distances between a set of query points and points.
    /*! \param query_points Query point positions.
        \param n_query_points The number of query points.
        \param points Point positions.
        \param n_points The number of points.
        \param distances Pointer to array of length n_query_points containing distances between each point and
       query_point (overwritten in place).
    */
    void computeDistances(const vec3<float>* query_points, const unsigned int n_query_points,
                          const vec3<float>* points, const unsigned int n_points, float* distances) const
    {
        if (n_query_points != n_points)
        {
            throw std::invalid_argument("The number of query points and points must match.");
        }
        util::forLoopWrapper(0, n_query_points, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                distances[i] = computeDistance(query_points[i], points[i]);
            }
        });
    }

    //! Calculate all pairwise distances between a set of query points and points.
    /*! \param query_points Query point positions.
        \param n_query_points The number of query points.
        \param points Point positions.
        \param n_points The number of points.
        \param distances Pointer to array of length n_query_points*n_points containing distances between
       points and query_points (overwritten in place).
    */
    void computeAllDistances(const vec3<float>* query_points, const unsigned int n_query_points,
                             const vec3<float>* points, const unsigned int n_points, float* distances) const
    {
        util::forLoopWrapper2D(
            0, n_query_points, 0, n_points, [&](size_t begin_n, size_t end_n, size_t begin_m, size_t end_m) {
                for (size_t i = begin_n; i < end_n; ++i)
                {
                    for (size_t j = begin_m; j < end_m; ++j)
                    {
                        distances[i * n_points + j] = computeDistance(query_points[i], points[j]);
                    }
                }
            });
    }

    //! Get mask of points that fit inside the box.
    /*! \param points Point positions.
        \param n_points The number of points.
        \param contains_mask Mask of points inside the box.
    */
    void contains(const vec3<float>* points, const unsigned int n_points, bool* contains_mask) const
    {
        util::forLoopWrapper(0, n_points, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < n_points; ++i)
            {
                std::transform(&points[begin], &points[end], &contains_mask[begin],
                               [this](const vec3<float>& point) -> bool {
                                   vec3<int> image(0, 0, 0);
                                   getImage(point, image);
                                   return image == vec3<int>(0, 0, 0);
                               });
            }
        });
    }

    //! Get the shortest distance between opposite boundary planes of the box
    /*! The distance between two planes of the lattice is 2 Pi/|b_i|, where
     *   b_1 is the reciprocal lattice vector of the Bravais lattice normal to
     *   the lattice vectors a_2 and a_3 etc.
     *
     * \return A vec3<float> containing the distance between the a_2-a_3, a_3-a_1 and
     *         a_1-a_2 planes for the triclinic lattice
     */
    vec3<float> getNearestPlaneDistance() const
    {
        vec3<float> dist;
        dist.x = m_L.x / std::sqrt(float(1.0) + m_xy * m_xy + (m_xy * m_yz - m_xz) * (m_xy * m_yz - m_xz));
        dist.y = m_L.y / std::sqrt(float(1.0) + m_yz * m_yz);
        dist.z = m_L.z;

        return dist;
    }

    /*! Get the lattice vector with index i
     *  \param i Index (0<=i<d) of the lattice vector, where d is dimension (2 or 3)
     *  \returns the lattice vector with index i
     */
    vec3<float> getLatticeVector(unsigned int i) const
    {
        if (i == 0)
        {
            return vec3<float>(m_L.x, 0.0, 0.0);
        }
        if (i == 1)
        {
            return vec3<float>(m_L.y * m_xy, m_L.y, 0.0);
        }
        if (i == 2 && !m_2d)
        {
            return vec3<float>(m_L.z * m_xz, m_L.z * m_yz, m_L.z);
        }
        throw std::out_of_range("Box lattice vector index requested does not exist.");
    }

    //! Set the periodic flags
    /*! \param periodic Flags to set
     *  \post Period flags are set to \a periodic
     */
    void setPeriodic(vec3<bool> periodic)
    {
        m_periodic = periodic;
    }

    void setPeriodic(bool x, bool y, bool z)
    {
        m_periodic = vec3<bool>(x, y, z);
    }

    //! Set the periodic flag along x
    void setPeriodicX(bool x)
    {
        m_periodic = vec3<bool>(x, m_periodic.y, m_periodic.z);
    }

    //! Set the periodic flag along y
    void setPeriodicY(bool y)
    {
        m_periodic = vec3<bool>(m_periodic.x, y, m_periodic.z);
    }

    //! Set the periodic flag along z
    void setPeriodicZ(bool z)
    {
        m_periodic = vec3<bool>(m_periodic.x, m_periodic.y, z);
    }

    //! Get the periodic flags
    vec3<bool> getPeriodic() const
    {
        return {m_periodic.x, m_periodic.y, m_periodic.z};
    }

    //! Get the periodic flag along x
    bool getPeriodicX() const
    {
        return m_periodic.x;
    }

    //! Get the periodic flag along y
    bool getPeriodicY() const
    {
        return m_periodic.y;
    }

    //! Get the periodic flag along z
    bool getPeriodicZ() const
    {
        return m_periodic.z;
    }

    void enforce2D() const
    {
        if (!is2D())
        {
            throw std::invalid_argument("A 3D box was provided to a class that only supports 2D systems.");
        }
    }

    void enforce3D() const
    {
        if (is2D())
        {
            throw std::invalid_argument("A 2D box was provided to a class that only supports 3D systems.");
        }
    }

private:
    vec3<float> m_lo;      //!< Minimum coords in the box
    vec3<float> m_hi;      //!< Maximum coords in the box
    vec3<float> m_L;       //!< L precomputed (used to avoid subtractions in boundary conditions)
    vec3<float> m_Linv;    //!< 1/L precomputed (used to avoid divisions in boundary conditions)
    float m_xy;            //!< xy tilt factor
    float m_xz;            //!< xz tilt factor
    float m_yz;            //!< yz tilt factor
    vec3<bool> m_periodic; //!< 0/1 to determine if the box is periodic in each direction
    bool m_2d;             //!< Specify whether box is 2D.
};

}; }; // end namespace freud::box

#endif // BOX_H
