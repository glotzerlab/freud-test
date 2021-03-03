// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef MATCH_ENV_H
#define MATCH_ENV_H

#include <map>
#include <vector>

#include "BiMap.h"
#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "Registration.h"
#include "VectorMath.h"

/*! \file MatchEnv.h
    \brief Particle environment matching
*/

namespace freud { namespace environment {
//! Clusters particles according to whether their local environments match or not, according to various shape
//! matching metrics

//! My environment data structure
struct Environment
{
    //! Constructor.
    Environment(bool ghost = false) : ghost(ghost) {}

    //! Add a vector to define the local environment
    void addVec(const vec3<float>& vec)
    {
        vecs.push_back(vec);
        vec_ind.push_back(num_vecs);
        num_vecs++;
    }

    unsigned int env_ind {0};      //!< The index of the environment
    std::vector<vec3<float>> vecs; //!< The vectors that define the environment
    //! Is this environment a ghost? Do we ignore it when we compute actual
    //  physical quantities associated with all environments?
    bool ghost;
    unsigned int num_vecs {0}; //!< The number of vectors currently defining the environment
    //! The order that the vectors must be in to define the environment
    std::vector<unsigned int> vec_ind;
    //! The rotation that defines the proper orientation of the environment
    rotmat3<float> proper_rot {};
};

//! General disjoint set class, taken mostly from Cluster.h
struct EnvDisjointSet
{
    //! Constructor (taken partially from Cluster.cc).
    explicit EnvDisjointSet(unsigned int Np);
    //! Merge two sets
    /*! Merge the two sets that elements a and b belong to. Taken partially
     * from Cluster.cc. The vec_map must be a bimap of PROPERLY ORDERED vector
     * indices where those of set a are on the left and those of set b are on
     * the right. The rotation must take the set of PROPERLY ROTATED vectors b
     * and rotate them to match the set of PROPERLY ROTATED vectors a
     */
    void merge(const unsigned int a, const unsigned int b, BiMap<unsigned int, unsigned int> vec_map,
               rotmat3<float>& rotation);

    //! Find the set with a given element (taken mostly from Cluster.cc).
    unsigned int find(const unsigned int c);

    //! Return ALL nodes in the tree that correspond to the head index m
    /*! Return ALL nodes in the tree that correspond to the head index m.
     * Values returned: the actual locations of the nodes in s. (i.e. if i is
     * returned, the node is accessed by s[i]). If environment m doesn't exist
     * as a HEAD in the set, throw an error.
     */
    std::vector<unsigned int> findSet(const unsigned int m);

    //! Get the vectors corresponding to environment head index m.
    /*! Vectors are averaged over all members of the environment cluster. Get
     * the vectors corresponding to environment head index m. Vectors are
     * averaged over all members of the environment cluster. If environment m
     * doesn't exist as a HEAD in the set, throw an error.
     */
    std::vector<vec3<float>> getAvgEnv(const unsigned int m);

    //! Get the vectors corresponding to index m in the dj set (throw an error if it doesn't exist).
    std::vector<vec3<float>> getIndividualEnv(const unsigned int m);

    std::vector<Environment> s;     //!< The disjoint set data
    std::vector<unsigned int> rank; //!< The rank of each tree in the set
    unsigned int m_max_num_neigh;   //!< The maximum number of neighbors in any environment in the set
};

/*****************************************************************************
 * There are various registration functions that are used by EnvironmentCluster but do *
 * not need to be exposed, or at least are not stateful and need not be      *
 * attached to the class.                                                    *
 *****************************************************************************/
//! Make environments from two sets of points.
/*! This function is primarily used to offer a simpler Python interface.
 */
std::pair<Environment, Environment> makeEnvironments(const box::Box& box, const vec3<float>* refPoints1,
                                                     vec3<float>* refPoints2, unsigned int numRef);

// Get the somewhat-optimal RMSD between the (PROPERLY REGISTERED) environment e1 and the (PROPERLY
// REGISTERED) environment e2.
/*! This function returns an std::pair of the rotation matrix that takes
 * the vectors of e2 to the vectors of e1 AND the mapping between the
 * properly indexed vectors of the environments that gives this RMSD.  NOTE
 * that this does not guarantee an absolutely minimal RMSD. It doesn't
 * figure out the optimal permutation of BOTH sets of vectors to minimize
 * the RMSD. Rather, it just figures out the optimal permutation of the
 * second set, the vector set used in the argument below. To fully solve
 * this, we need to use the Hungarian algorithm or some other way of
 * solving the so-called assignment problem.
 *
 * \param e1 First environment.
 * \param e2 First environment.
 * \param min_rmsd The value of the minimum RMSD (updated by reference).
 * \param registration Controls whether we first use brute force registration to
 *                     orient the second set of vectors such that it
 *                     minimizes the RMSD between the two sets
 */
std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>> minimizeRMSD(Environment& e1, Environment& e2,
                                                                          float& min_rmsd, bool registration);

//! Overload of the above minimizeRMSD function that provides an easier interface to Python.
/*! Construct the environments accordingly, and utilize minimizeRMSD() as
 * above. Arguments are pointers to interface directly with python. Return
 * a std::map (for ease of use) with the mapping between vectors refPoints1
 * and refPoints2 that gives this RMSD.
 *
 * WARNING: If registration=True, then refPoints2 is CHANGED by this function.
 *
 * \param refPoints1 Points composing first environment.
 * \param refPoints2 Points composing second environment.
 * \param numRef Number of points.
 * \param min_rmsd The value of the minimum RMSD (updated by reference).
 * \param registration Controls whether we first use brute force registration to
 *                     orient the second set of vectors such that it
 *                     minimizes the RMSD between the two sets
 */
std::map<unsigned int, unsigned int> minimizeRMSD(const box::Box& box, const vec3<float>* refPoints1,
                                                  vec3<float>* refPoints2, unsigned int numRef,
                                                  float& min_rmsd, bool registration);

//! Finds a rotation matrix and the appropriate index correspondence between two environments if it exists.
/*! If the two environments correspond, returns a std::pair of the rotation matrix that takes the
 * vectors of e2 to the vectors of e1 AND the mapping between the properly
 * indexed vectors of the environments that will make them correspond to
 * each other. If not, return a std::pair of the identity matrix AND an
 * empty map.
 *
 * \param e1 First environment.
 * \param e2 First environment.
 * \param threshold_sq This quantity is the square of the maximum magnitude of
 *                     the vector difference between two vectors, below which
 *                     you call them matching. Recommended values for the
 *                     threshold are 10-30% of the first minimum of the radial
 *                     distribution function (so the argument should be the
 *                     square of that).
 * \param registration Controls whether we first use brute force registration to
 *                     orient the second set of vectors such that it
 *                     minimizes the RMSD between the two sets
 */
std::pair<rotmat3<float>, BiMap<unsigned int, unsigned int>> isSimilar(Environment& e1, Environment& e2,
                                                                       float threshold_sq, bool registration);

//! Overload of the above isSimilar function that provides an easier interface to Python.
/*! If the two environments correspond, returns a std::pair of the rotation matrix that takes the
 * vectors of e2 to the vectors of e1 AND the mapping between the properly
 * indexed vectors of the environments that will make them correspond to
 * each other. If not, return a std::pair of the identity matrix AND an
 * empty map.
 *
 * WARNING: If registration=True, then refPoints2 is CHANGED by this function.
 *
 * \param refPoints1 Points composing first environment.
 * \param refPoints2 Points composing second environment.
 * \param numRef Number of points.
 * \param threshold_sq This quantity is the square of the maximum magnitude of the vector
 *                     difference between two vectors, below which you call
 *                     them matching. Recommended values for the threshold are
 *                     10-30% of the first minimum of the radial distribution
 *                     function (so the argument should be the square of that).
 * \param registration Controls whether we first use brute force registration to
 *                     orient the second set of vectors such that it
 *                     minimizes the RMSD between the two sets
 */
std::map<unsigned int, unsigned int> isSimilar(const box::Box& box, const vec3<float>* refPoints1,
                                               vec3<float>* refPoints2, unsigned int numRef,
                                               float threshold_sq, bool registration);

//! Parent class for environment matching.
/*! This class defines some of the common features of the different environment
 * matching classes. All of them perform some form of registration for
 * matching, but the precise feature set differs between the classes.
 */
class MatchEnv
{
public:
    MatchEnv();

    ~MatchEnv();

    //! Construct and return a local environment surrounding the particle indexed by i. Set the environment
    //! index to env_ind.
    static Environment buildEnv(const freud::locality::NeighborQuery* nq,
                                const freud::locality::NeighborList* nlist, size_t num_bonds, size_t& bond,
                                unsigned int i, unsigned int env_ind);

    //! Returns the entire Np by m_num_neighbors by 3 matrix of all environments for all particles
    const util::ManagedArray<vec3<float>>& getPointEnvironments()
    {
        return m_point_environments;
    }

protected:
    util::ManagedArray<vec3<float>> m_point_environments; //!< m_NP by m_max_num_neighbors by 3 matrix of all
                                                          //!< environments for all particles
};

//! Cluster particles with similar environments.
/*! The environment matching method is defined according to the paper "Identity
 * crisis in alchemical space drives the entropic colloidal glass transition"
 * by Erin G. Teich (http://dx.doi.org/10.1038/s41467-018-07977-2). The core of
 * the method is a brute force point set registration of the bonds betweena
 * point and its nearest neighbors with the corresponding bonds of another
 * point. By performing this sort of registration between various pairs of
 * points, we identify regions where neighboring points share similar local
 * environments.
 */
class EnvironmentCluster : public MatchEnv
{
public:
    //! Constructor
    /*!
     * \param box The system box.
     * \param num_neighbors Number of nearest neighbors taken to define the local environment of any given
     * particle.
     */
    EnvironmentCluster() : MatchEnv() {}

    //! Destructor
    ~EnvironmentCluster();

    //! Determine clusters of particles with matching environments
    /*! This is the primary interface to EnvironmentCluster. It computes particle
     * environments and then attempts to cluster nearby particles with similar
     * environments, unless global is set true. Otherwise, it performs a
     * pairwise comparison of all particle environments to perform the match.
     * WARNING: A global search can be extremely slow.
     * This is taken from Cluster.cc and SolLiq.cc and LocalQlNear.cc
     *
     * \param env_nlist The NeighborList used to build the environment of every particle.
     * \param nlist The NeighborList used to determine the neighbors against which
     *              to compare environments for every particle, if hard_r = False.
     * \param threshold This quantity is of the maximum magnitude of the
     *                  vector difference between two vectors, below which
     *                  you call them matching. Recommended values for the
     *                  threshold are 10-30% of the first minimum of the radial
     *                  distribution function.
     * \param registration Controls whether we first use brute force registration to
     *                     orient the second set of vectors such that it
     *                     minimizes the RMSD between the two sets
     * \param global If true, do an exhaustive search wherein you compare the
     *               environments of every single pair of particles in the
     *               simulation. If global is false, only compare the
     *               environments of neighboring particles.
     */
    void compute(const freud::locality::NeighborQuery* nq, const freud::locality::NeighborList* nlist_arg,
                 locality::QueryArgs qargs, const freud::locality::NeighborList* env_nlist_arg,
                 locality::QueryArgs env_qargs, float threshold, bool registration = false,
                 bool global = false);

    //! Get a reference to the particles, indexed into clusters according to their matching local environments
    const util::ManagedArray<unsigned int>& getClusters()
    {
        return m_env_index;
    }

    //! Get a reference to the particles, indexed into clusters according to their matching local environments
    std::vector<std::vector<vec3<float>>> getClusterEnvironments()
    {
        return m_cluster_environments;
    }

    unsigned int getNumClusters() const
    {
        return m_num_clusters;
    }

private:
    //! Populate the env_index, env and tot_env arrays (updated by reference) based on the contents of an
    //! EnvDisjointSet.
    /*! T
     * \param dj The object encoding the set of clusters found.
     * \param env_index An array indexed by point id indicating the cluster id for each point.
     * \param cluster_env A mapping from cluster id to a list of vectors composing
     *                    that cluster's environment. A null pointer may be passed
     *                    indicating that we do not need to keep track of cluster
     *                    environments.
     * \param tot_env An array indexed by point id indicating the environment of each point.
     * \param reLabel If true, the cluster ids are relabeled to ensur contiguous
     *                ordering from 0 to the number of clusters. Otherwise, the
     *                cluster ids will match the point id of the first point found
     *                with that environment (which defines the cluster).
     * \return The number of clusters found.
     */
    unsigned int populateEnv(EnvDisjointSet dj);

    unsigned int m_num_clusters {0};              //!< Last number of local environments computed
    util::ManagedArray<unsigned int> m_env_index; //!< Cluster index determined for each particle
    std::vector<std::vector<vec3<float>>>
        m_cluster_environments; //!< Dictionary of (cluster id, vectors) pairs
};

//! Match local point environments to a specific motif.
/*! The environment matching method is defined according to the paper "Identity
 * crisis in alchemical space drives the entropic colloidal glass transition"
 * by Erin G. Teich (http://dx.doi.org/10.1038/s41467-018-07977-2). This class
 * is primarily provided as a companion to EnvironmentCluster that can be used
 * to more closely analyze specific motifs. Rather than clustering all points
 * in a system based on their local environments, this class provides more a
 * more fine-grained computation to analyze which points match a specified
 * motif.
 */
class EnvironmentMotifMatch : public MatchEnv
{
public:
    //! Constructor
    /*!
     * \param box The system box.
     * \param num_neighbors Number of nearest neighbors taken to define the local environment of any given
     * particle.
     */
    EnvironmentMotifMatch() : MatchEnv() {}

    //! Determine whether particles match a given input motif.
    /*! Given a motif composed of vectors that represent the vectors connecting
     * a point to the neighbors that are part of the motif, matchMotif looks at
     * every point in points and checks if its neighbors may match this motif.
     * Any point whose local environment matches the motif is marked as part of
     * cluster 0 in the clusters array. All others are ignored. The
     * tot_environments array is updated with the vectors composing the
     * environment of every particle.
     *
     * \param nlist A NeighborList instance.
     * \param points The points to test against the motif.
     * \param Np The number of points.
     * \param motif The vectors characterizing the motif. Note that these are
     *              vectors, so for instance given a square motif composed of
     *              points at the corners of a square, the motif should not
     *              include the point (0, 0) because what we are matching is
     *              the vectors to the neighbors.
     * \param motif_size The number of vectors characterizing the motif.
     * \param threshold This quantity is of the maximum magnitude of the
     *                  vector difference between two vectors, below which
     *                  you call them matching. Recommended values for the
     *                  threshold are 10-30% of the first minimum of the radial
     *                  distribution function.
     * \param registration Controls whether we first use brute force registration to
     *                     orient the second set of vectors such that it
     *                     minimizes the RMSD between the two sets
     */
    void compute(const freud::locality::NeighborQuery* nq, const freud::locality::NeighborList* nlist_arg,
                 locality::QueryArgs qargs, const vec3<float>* motif, unsigned int motif_size,
                 float threshold, bool registration = false);

    //! Return the array indicating whether each particle matched the motif or not.
    const util::ManagedArray<bool>& getMatches()
    {
        return m_matches;
    }

private:
    util::ManagedArray<bool>
        m_matches; //!< Boolean array indicating whether or not a particle's environment matches the motif.
};

//! Compute RMSDs of the local particle environments.
/*! The environment matching method is defined according to the paper "Identity
 * crisis in alchemical space drives the entropic colloidal glass transition"
 * by Erin G. Teich (http://dx.doi.org/10.1038/s41467-018-07977-2). Similar to
 * EnvironmentMotifMatch, this class is primarily provided as a companion to
 * EnvironmentCluster that can be used to more closely analyze specific motifs.
 * The purpose of this class is to find the optimal transformation that maps a
 * set of points onto a specified motif.
 */
class EnvironmentRMSDMinimizer : public MatchEnv
{
public:
    //! Constructor
    /*!
     * \param num_neighbors Number of nearest neighbors taken to define the local environment of any given
     * particle.
     */
    EnvironmentRMSDMinimizer() : MatchEnv() {}

    //! Rotate (if registration=True) and permute the environments of all particles to minimize their RMSD wrt
    //! a given input motif.
    /*! Returns a vector of minimal RMSD values, one value per particle. NOTE
     * that this does not guarantee an absolutely minimal RMSD. It doesn't
     * figure out the optimal permutation of BOTH sets of vectors to minimize
     * the RMSD.  Rather, it just figures out the optimal permutation of the
     * second set, the vector set used in the argument below. To fully solve
     * this, we need to use the Hungarian algorithm or some other way of
     * solving the so-called assignment problem.
     *
     * \param nlist A NeighborList instance.
     * \param points The points to test against the motif.
     * \param Np The number of points.
     * \param motif The vectors characterizing the motif. Note that these are
     *              vectors, so for instance given a square motif composed of
     *              points at the corners of a square, the motif should not
     *              include the point (0, 0) because what we are matching is
     *              the vectors to the neighbors.
     * \param motif_size The number of vectors characterizing the motif.
     * \param threshold This quantity is of the maximum magnitude of the
     *                  vector difference between two vectors, below which
     *                  you call them matching. Recommended values for the
     *                  threshold are 10-30% of the first minimum of the radial
     *                  distribution function.
     * \param registration Controls whether we first use brute force registration to
     *                     orient the second set of vectors such that it
     *                     minimizes the RMSD between the two sets
     */
    void compute(const freud::locality::NeighborQuery* nq, const freud::locality::NeighborList* nlist_arg,
                 locality::QueryArgs qargs, const vec3<float>* motif, unsigned int motif_size,
                 bool registration = false);

    //! Return the array indicating whether or not a successful mapping was found between each particle and
    //! the provided motif.
    const util::ManagedArray<float>& getRMSDs()
    {
        return m_rmsds;
    }

private:
    util::ManagedArray<float>
        m_rmsds; //!< Boolean array indicating whether or not a particle's environment matches the motif.
};

}; }; // end namespace freud::environment

#endif // MATCH_ENV_H
