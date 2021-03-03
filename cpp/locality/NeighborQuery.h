// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEIGHBOR_QUERY_H
#define NEIGHBOR_QUERY_H

#include <memory>
#include <stdexcept>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_sort.h>
#include <utility>

#include "Box.h"
#include "NeighborBond.h"
#include "NeighborList.h"
#include "NeighborPerPointIterator.h"
#include "utils.h"

/*! \file NeighborQuery.h
    \brief Defines the abstract API for collections of points that can be
           queried against for neighbors.
*/

namespace freud { namespace locality {

//! Enumeration for types of queries.
enum QueryType
{
    none,    //! Default query type to avoid implicit default types.
    ball,    //! Query based on distance cutoff.
    nearest, //! Query based on number of requested neighbors.
};

constexpr auto DEFAULT_MODE = QueryType::none;            //!< Default mode.
constexpr unsigned int DEFAULT_NUM_NEIGHBORS(0xffffffff); //!< Default number of neighbors.
constexpr float DEFAULT_R_MAX(-1.0);                      //!< Default maximum query distance.
constexpr float DEFAULT_R_MIN(0);                         //!< Default minimum query distance.
constexpr float DEFAULT_R_GUESS(-1.0);                    //!< Default guess query distance.
constexpr float DEFAULT_SCALE(-1.0);      //!< Default scaling parameter for AABB nearest neighbor queries.
constexpr bool DEFAULT_EXCLUDE_II(false); //!< Default for whether or not to include self-neighbors.
constexpr auto ITERATOR_TERMINATOR
    = NeighborBond(-1, -1, 0); //!< The object returned when iteration is complete.

//! POD class to hold information about generic queries.
/*! This class provides a standard method for specifying the type of query to
 *  perform with a NeighborQuery object. Rather than calling queryBall
 *  specifically, for example, the user can call a generic querying function and
 *  provide an instance of this class to specify the nature of the query.
 */
struct QueryArgs
{
    QueryArgs() = default;

    QueryType mode {DEFAULT_MODE}; //! Whether to perform a ball or k-nearest neighbor query.
    unsigned int num_neighbors {DEFAULT_NUM_NEIGHBORS}; //! The number of nearest neighbors to find.
    float r_max {DEFAULT_R_MAX};          //! The cutoff distance within which to find neighbors.
    float r_min {DEFAULT_R_MIN};          //! The minimum distance beyond which to find neighbors.
    float r_guess {DEFAULT_R_GUESS};      //! The initial distance for finding neighbors, used by some
                                          //! algorithms to initialize a number of neighbors query.
    float scale {DEFAULT_SCALE};          //! The scale factor to use when performing repeated ball queries
                                          //! to find a specified number of nearest neighbors.
    bool exclude_ii {DEFAULT_EXCLUDE_II}; //! If true, exclude self-neighbors.
};

// Forward declare the iterators
class NeighborQueryIterator;
class NeighborQueryPerPointIterator;

//! Parent data structure for all neighbor finding algorithms.
/*! This class defines the API for all data structures for accelerating
 *  neighbor finding. The object encapsulates a set of points and a system box
 *  that define the set of points to search and the periodic system within these
 *  points can be found. The interface for finding neighbors is the query
 *  method, which generates an iterator that finds all requested neighbors.
 */
class NeighborQuery
{
public:
    //! Nullary constructor for Cython
    NeighborQuery() = default;

    //! Constructor
    NeighborQuery(box::Box box, const vec3<float>* points, unsigned int n_points)
        : m_box(std::move(box)), m_points(points), m_n_points(n_points)
    {
        // Reject systems with 0 particles
        if (m_n_points == 0)
        {
            throw std::invalid_argument("Cannot create a NeighborQuery with 0 particles.");
        }

        // For 2D systems, check if any z-coordinates are outside some tolerance of z=0
        if (m_box.is2D())
        {
            for (unsigned int i(0); i < n_points; i++)
            {
                if (std::abs(m_points[i].z) > 1e-6)
                {
                    throw std::invalid_argument("A point with z != 0 was provided in a 2D box.");
                }
            }
        }
    }

    //! Empty Destructor
    virtual ~NeighborQuery() = default;

    //! Perform a query based on a set of query parameters.
    /*! Given a QueryArgs object and a set of points to perform a query
     *  with, this function creates an iterator object that loops over all \c
     *  query_points and returns their neighbors. Specific query logic is
     *  implemented on a per-particle basis through the querySingle function,
     *  which should be overriden by subclasses to apply the correct neighbor
     *  finding logic.
     *
     *  \param query_points The points to find neighbors for.
     *  \param n_query_points The number of query points.
     *  \param qargs The query arguments that should be used to find neighbors.
     */
    virtual std::shared_ptr<NeighborQueryIterator>
    query(const vec3<float>* query_points, unsigned int n_query_points, QueryArgs query_args) const
    {
        // pair calculations using non-periodic boxes should fail
        vec3<bool> periodic = m_box.getPeriodic();
        if (!(periodic.x && periodic.y && periodic.z))
        {
            throw std::domain_error("Pair queries in a non-periodic box are not implemented.");
        }

        this->validateQueryArgs(query_args);
        return std::make_shared<NeighborQueryIterator>(this, query_points, n_query_points, query_args);
    }

    //! Perform a per-particle query based on a set of query parameters.
    /*! This function is the primary interface by which subclasses provide
     *  logic for finding neighbors. All such logic should be contained in
     *  subclasses of the NeighborQueryPerPointIterator that can then generate
     *  neighbors on-the-fly.
     *
     *  \param query_point The point to find neighbors for.
     *  \param n_query_points The number of query points.
     *  \param qargs The query arguments that should be used to find neighbors.
     */
    virtual std::shared_ptr<NeighborQueryPerPointIterator>
    querySingle(const vec3<float> query_point, unsigned int query_point_idx, QueryArgs args) const = 0;

    //! Get the simulation box
    const box::Box& getBox() const
    {
        return m_box;
    }

    //! Get the reference points
    const vec3<float>* getPoints() const
    {
        return m_points;
    }

    //! Get the number of reference points
    unsigned int getNPoints() const
    {
        return m_n_points;
    }

    //! Get a point's coordinates using index operator notation
    /*! \param index The point index to return.
     */
    vec3<float> operator[](unsigned int index) const
    {
        if (index >= m_n_points)
        {
            throw std::runtime_error("NeighborQuery attempted to access a point with index >= n_points.");
        }
        return m_points[index];
    }

protected:
    //! Validate the combination of specified arguments.
    /*! Before checking if the combination of parameters currently set is
     *  valid, this function first attempts to infer a mode if one is not set in
     *  order to allow the user to specify certain simple minimal argument
     *  combinations (e.g. just an r_max) without having to specify the mode
     *  explicitly.
     */
    virtual void validateQueryArgs(QueryArgs& args) const
    {
        inferMode(args);
        // Validate remaining arguments.
        if (args.mode == QueryType::ball)
        {
            if (args.r_max == DEFAULT_R_MAX)
            {
                throw std::runtime_error(
                    "You must set r_max in the query arguments when performing ball queries.");
            }
            if (args.num_neighbors != DEFAULT_NUM_NEIGHBORS)
            {
                throw std::runtime_error(
                    "You cannot set num_neighbors in the query arguments when performing ball queries.");
            }
        }
        else if (args.mode == QueryType::nearest)
        {
            if (args.num_neighbors == DEFAULT_NUM_NEIGHBORS)
            {
                throw std::runtime_error("You must set num_neighbors in the query arguments when performing "
                                         "number of neighbor queries.");
            }
            if (args.r_max == DEFAULT_R_MAX)
            {
                args.r_max = std::numeric_limits<float>::infinity();
            }
        }
        else
        {
            throw std::runtime_error("Unknown mode");
        }
    }

    //! Try to determine the query mode if one is not specified.
    /*! If no mode is specified and a number of neighbors is specified, the
     *  query mode must be a nearest neighbors query (all other arguments can
     *  reasonably modify that query). Otherwise, if a max distance is set we
     *  can assume a ball query is desired.
     */
    virtual void inferMode(QueryArgs& args) const
    {
        // Infer mode if possible.
        if (args.mode == QueryType::none)
        {
            if (args.num_neighbors != DEFAULT_NUM_NEIGHBORS)
            {
                args.mode = QueryType::nearest;
            }
            else if (args.r_max != DEFAULT_R_MAX)
            {
                args.mode = QueryType::ball;
            }
        }
    }

    const box::Box m_box;        //!< Simulation box where the particles belong.
    const vec3<float>* m_points; //!< Point coordinates.
    unsigned int m_n_points;     //!< Number of points.
};

//! Implementation of per-point finding logic for NeighborQuery objects.
/*! This abstract class specializes a few of the methods of its parent for the
 *  case of working with NeighborQuery objects. In particular, on construction
 *  it takes information on the NeighborQuery object it is attached to, allowing
 *  iterators to access the points the NeighborQuery was constructed for.
 *  Additionally, it defines the standard interface by which such iterations
 *  should indicate that all neighbors have been found, which is by setting the
 *  m_finished flag.
 */
class NeighborQueryPerPointIterator : public NeighborPerPointIterator
{
public:
    //! Nullary constructor for Cython
    NeighborQueryPerPointIterator() = default;

    //! Constructor
    NeighborQueryPerPointIterator(const NeighborQuery* neighbor_query, const vec3<float>& query_point,
                                  unsigned int query_point_idx, float r_max, float r_min, bool exclude_ii)
        : NeighborPerPointIterator(query_point_idx), m_neighbor_query(neighbor_query),
          m_query_point(query_point), m_finished(false), m_r_max(r_max), m_r_min(r_min),
          m_exclude_ii(exclude_ii)
    {}

    //! Empty Destructor
    ~NeighborQueryPerPointIterator() override = default;

    //! Indicate when done.
    bool end() const override
    {
        return m_finished;
    }

    //! Get the next element.
    NeighborBond next() override = 0;

protected:
    const NeighborQuery* m_neighbor_query;       //!< Link to the NeighborQuery object.
    const vec3<float> m_query_point = {0, 0, 0}; //!< Coordinates of the query point.
    bool m_finished; //!< Flag to indicate that iteration is complete (must be set by next() on termination).
    float m_r_max;   //!< Cutoff distance for neighbors.
    float m_r_min;   //!< Minimum distance for neighbors.
    bool m_exclude_ii; //!< Flag to indicate whether or not to include self bonds.
};

//! The iterator class for neighbor queries on NeighborQuery objects.
/*! All queries to a NeighborQuery return instances of this class. The
 *  NeighborQueryIterator is capable of either iterating over all neighbors of
 *  the provided query_points based on the set of points contained in the
 *  NeighborQuery object, or of providing NeighborQueryPerPoint iterator
 *  instances for any of the query_points it was constructed with. The first
 *  interface is much more convenient for user interaction, while the second is
 *  primarily provided to support thread-safe parallelism. This class is not
 *  designed to be inherited from; the implementation of the querySingle method
 *  in NeighborQuery subclasses (to return per-point iterators) should be
 *  sufficient for this class to work.
 */
class NeighborQueryIterator
{
public:
    //! Constructor
    NeighborQueryIterator(const NeighborQuery* neighbor_query, const vec3<float>* query_points,
                          unsigned int num_query_points, QueryArgs& qargs)
        : m_neighbor_query(neighbor_query), m_query_points(query_points),
          m_num_query_points(num_query_points), m_qargs(qargs), m_finished(false), m_cur_p(0)
    {
        m_iter = this->query(m_cur_p);
    }

    //! Empty Destructor
    ~NeighborQueryIterator() = default;

    //! Indicate when done.
    bool end() const
    {
        return m_finished;
    }

    //! Get an iterator for a specific query point by index.
    std::shared_ptr<NeighborQueryPerPointIterator> query(unsigned int i)
    {
        return m_neighbor_query->querySingle(m_query_points[i], i, m_qargs);
    }

    //! Get the next element.
    NeighborBond next()
    {
        if (m_finished)
        {
            return ITERATOR_TERMINATOR;
        }
        NeighborBond nb;
        while (true)
        {
            while (!m_iter->end())
            {
                nb = m_iter->next();

                if (nb != ITERATOR_TERMINATOR)
                {
                    return nb;
                }
            }
            m_cur_p++;
            if (m_cur_p >= m_num_query_points)
            {
                break;
            }
            m_iter = this->query(m_cur_p);
        }
        m_finished = true;
        return ITERATOR_TERMINATOR;
    }

    //! Generate a NeighborList from query.
    /*! This function exploits parallelism by finding the neighbors for
     *  each query point in parallel and adding them to a list, which is
     *  then sorted in parallel as well before being added to the
     *  NeighborList object. Right now this won't be backwards compatible
     *  because the kn query is not symmetric, so even if we reverse the
     *  output order here the actual neighbors found will be different.
     *
     *  This function returns a pointer, not a shared pointer, so the
     *  caller is responsible for deleting it. The reason for this is that
     *  the primary use-case is to have this object be managed by instances
     *  of the Cython NeighborList class.
     */
    NeighborList* toNeighborList(bool sort_by_distance = false)
    {
        using BondVector = tbb::enumerable_thread_specific<std::vector<NeighborBond>>;
        BondVector bonds;
        util::forLoopWrapper(0, m_num_query_points, [&](size_t begin, size_t end) {
            BondVector::reference local_bonds(bonds.local());
            NeighborBond nb;
            for (size_t i = begin; i < end; ++i)
            {
                std::shared_ptr<NeighborQueryPerPointIterator> it = this->query(i);
                while (!it->end())
                {
                    nb = it->next();
                    // If we're excluding ii bonds, we have to check before adding.
                    if (nb != ITERATOR_TERMINATOR)
                    {
                        local_bonds.emplace_back(nb.query_point_idx, nb.point_idx, nb.distance);
                    }
                }
            }
        });

        tbb::flattened2d<BondVector> flat_bonds = tbb::flatten2d(bonds);
        std::vector<NeighborBond> linear_bonds(flat_bonds.begin(), flat_bonds.end());
        if (sort_by_distance)
        {
            tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end(), compareNeighborDistance);
        }
        else
        {
            tbb::parallel_sort(linear_bonds.begin(), linear_bonds.end(), compareNeighborBond);
        }

        unsigned int num_bonds = linear_bonds.size();

        auto* nl = new NeighborList();
        nl->setNumBonds(num_bonds, m_num_query_points, m_neighbor_query->getNPoints());

        util::forLoopWrapper(0, num_bonds, [&](size_t begin, size_t end) {
            for (size_t bond = begin; bond < end; ++bond)
            {
                nl->getNeighbors()(bond, 0) = linear_bonds[bond].query_point_idx;
                nl->getNeighbors()(bond, 1) = linear_bonds[bond].point_idx;
                nl->getDistances()[bond] = linear_bonds[bond].distance;
                nl->getWeights()[bond] = float(1.0);
            }
        });

        return nl;
    }

protected:
    const NeighborQuery* m_neighbor_query;                 //!< Link to the NeighborQuery object.
    const vec3<float>* m_query_points;                     //!< Coordinates of the query points.
    unsigned int m_num_query_points;                       //!< The number of query points.
    const QueryArgs m_qargs;                               //!< The query arguments
    std::shared_ptr<NeighborQueryPerPointIterator> m_iter; //!< The per-point iterator being used.

    bool m_finished; //!< Flag to indicate that iteration is complete (must be set by next on termination).
    unsigned int m_cur_p; //!< The current particle under consideration.
};

}; }; // end namespace freud::locality

#endif // NEIGHBOR_QUERY_H
