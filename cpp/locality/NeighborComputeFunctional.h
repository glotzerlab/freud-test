#ifndef NEIGHBOR_COMPUTE_FUNCTIONAL_H
#define NEIGHBOR_COMPUTE_FUNCTIONAL_H

#include <memory>

#include "AABBQuery.h"
#include "NeighborList.h"
#include "NeighborPerPointIterator.h"
#include "NeighborQuery.h"
#include "utils.h"

/*! \file NeighborComputeFunctional.h
    \brief Implements logic for generic looping over neighbors and applying a compute function.
*/

namespace freud { namespace locality {

//! Make a default NeighborList object to use.
/*! This function makes a NeighborList from the provided NeighborQuery object
 * if the provided NeighborList is NULL. Otherwise, it simply returns a copy of
 * the provided NeighborList.
 */
NeighborList makeDefaultNlist(const NeighborQuery* nq, const NeighborList* nlist,
                              const vec3<float>* query_points, unsigned int num_query_points,
                              locality::QueryArgs qargs);

//! Compute the vector corresponding to a NeighborBond.
/*! The primary purpose of this function is to standardize the directionality
 * of the delta vector, which is defined as pointing from the query_point to
 * the point (point - query_point), wrapped into the box.
 */
inline vec3<float> bondVector(const NeighborBond& nb, const NeighborQuery* nq,
                              const vec3<float>* query_points)
{
    return nq->getBox().wrap((*nq)[nb.point_idx] - query_points[nb.query_point_idx]);
}

//! Implementation of per-point finding logic for NeighborList objects.
/*! This class provides a concrete implementation of the per-point neighbor
 *  finding interface specified by the NeighborPerPointIterator. In particular,
 *  it includes the logic for finding neighbors within a NeighborList by using
 *  the find_first_index method to initialize a start index and looping over all
 *  neighbors in the NeighborList.
 */
class NeighborListPerPointIterator : public NeighborPerPointIterator
{
public:
    NeighborListPerPointIterator(const NeighborList* nlist, size_t point_index)
        : NeighborPerPointIterator(point_index), m_nlist(nlist)
    {
        m_current_index = m_nlist->find_first_index(point_index);
        m_finished = m_current_index == m_nlist->getNumBonds();
        if (!m_finished)
        {
            m_returned_point_index = m_nlist->getNeighbors()(m_current_index, 0);
        }
    }

    ~NeighborListPerPointIterator() override = default;

    NeighborBond next() override
    {
        if (m_current_index == m_nlist->getNumBonds())
        {
            m_finished = true;
            return ITERATOR_TERMINATOR;
        }

        NeighborBond nb = NeighborBond(
            m_nlist->getNeighbors()(m_current_index, 0), m_nlist->getNeighbors()(m_current_index, 1),
            m_nlist->getDistances()[m_current_index], m_nlist->getWeights()[m_current_index]);
        ++m_current_index;
        m_returned_point_index = nb.query_point_idx;
        return nb;
    }

    bool end() const override
    {
        return (m_returned_point_index != m_query_point_idx) || m_finished;
    }

private:
    const NeighborList* m_nlist; //! The NeighborList being iterated over.
    size_t m_current_index;      //! The row of m_nlist where the iterator is currently located.
    size_t m_returned_point_index {
        0xffffffff}; //! The index of the last returned point (i.e. the value of
                     //! m_nlist.getNeighbors()(m_current_index, 0)). Initialized to an arbitrary sentinel in
                     //! case the user ever tries to access this for an iterator built from an empty nlist.
    bool m_finished; //! Flag to indicate that the iterator has been exhausted.
};

//! Wrapper iterating looping over NeighborQuery or NeighborList.
/*! This function dynamically determines whether or not the provided
 *  NeighborList is valid. If it is, it applies the provide compute function to
 *  all neighbor pairs in the NeighborList. If not, it attempts to use the
 *  provided NeighborQuery for iteration. If the NeighborQuery object is also
 *  not queryable (if it's a RawPoints object), a local NeighborQuery instance
 *  is created and iterated over.
 *
 *  This function is designed for computations that must performs some sort of
 *  pre- or post-processing on a per-point basis. The provide compute function
 *  is called in a loop, once for each query_point, allowing the user to e.g.
 *  normalize some computed quantity based on the number of neighbor points
 *  found for a given query_point. As a result, the compute function must accept
 *  both a query_point index and a NeighborPerPointIterator that provides the
 *  neighbors of that query_point.
 *
 *  \param neighbor_query NeighborQuery object to iterate over.
 *  \param query_points Query points to perform computation on.
 *  \param n_query_points Number of query_points.
 *  \param qargs Query arguments.
 *  \param nlist Neighbor List. If not NULL, loop over it. Otherwise, use neighbor_query appropriately with
 * given qargs. \param cf An object with operator(size_t point_index, std::shared_ptr<NeighborIterator>) as
 * input. It should implement iteration logic over the iterator.
 */
template<typename ComputePairType>
void loopOverNeighborsIterator(const NeighborQuery* neighbor_query, const vec3<float>* query_points,
                               unsigned int n_query_points, QueryArgs qargs, const NeighborList* nlist,
                               const ComputePairType& cf, bool parallel = true)
{
    // check if nlist exists
    if (nlist != nullptr)
    {
        util::forLoopWrapper(
            0, n_query_points,
            [=](size_t begin, size_t end) {
                for (size_t i = begin; i != end; ++i)
                {
                    std::shared_ptr<NeighborListPerPointIterator> niter
                        = std::make_shared<NeighborListPerPointIterator>(nlist, i);
                    cf(i, niter);
                }
            },
            parallel);
    }
    else
    {
        std::shared_ptr<NeighborQueryIterator> iter
            = neighbor_query->query(query_points, n_query_points, qargs);

        // iterate over the query object in parallel
        util::forLoopWrapper(
            0, n_query_points,
            [=](size_t begin, size_t end) {
                for (size_t i = begin; i != end; ++i)
                {
                    std::shared_ptr<NeighborQueryPerPointIterator> it = iter->query(i);
                    cf(i, it);
                }
            },
            parallel);
    }
}

//! Wrapper iterating looping over NeighborQuery or NeighborList.
/*! This function dynamically determines whether or not the provided
 *  NeighborList is valid. If it is, it applies the provide compute function to
 *  all neighbor pairs in the NeighborList. If not, it attempts to use the
 *  provided NeighborQuery for iteration. If the NeighborQuery object is also
 *  not queryable (if it's a RawPoints object), a local NeighborQuery instance
 *  is created and iterated over.
 *
 *  This function is designed for computations that can simplify accumulate
 *  over all neighbor pairs. As a result, the provided compute function is
 *  simply applied to all pairs, allowing maximum parallelism. The actual logic
 *  for the NeighborList vs NeighborQuery code paths are handled in helper
 *  functions.
 *
 *  \param neighbor_query NeighborQuery object to iterate over.
 *  \param query_points Query points to perform computation on.
 *  \param n_query_points Number of query_points.
 *  \param qargs Query arguments.
 *  \param nlist Neighbor List. If not NULL, loop over it. Otherwise, use neighbor_query appropriately with
 * given qargs. \param cf An object with operator(NeighborBond) as input.
 */
template<typename ComputePairType>
void loopOverNeighbors(const NeighborQuery* neighbor_query, const vec3<float>* query_points,
                       unsigned int n_query_points, QueryArgs qargs, const NeighborList* nlist,
                       const ComputePairType& cf, bool parallel = true)
{
    // check if nlist exists
    if (nlist != nullptr)
    {
        util::forLoopWrapper(
            0, nlist->getNumBonds(),
            [=](size_t begin, size_t end) {
                for (size_t bond = begin; bond != end; ++bond)
                {
                    const NeighborBond nb(nlist->getNeighbors()(bond, 0), nlist->getNeighbors()(bond, 1),
                                          nlist->getDistances()[bond], nlist->getWeights()[bond]);
                    cf(nb);
                }
            },
            parallel);
    }
    else
    {
        std::shared_ptr<NeighborQueryIterator> iter
            = neighbor_query->query(query_points, n_query_points, qargs);

        // iterate over the query object in parallel
        util::forLoopWrapper(
            0, n_query_points,
            [&iter, &cf](size_t begin, size_t end) {
                NeighborBond nb;
                for (size_t i = begin; i != end; ++i)
                {
                    std::shared_ptr<NeighborQueryPerPointIterator> it = iter->query(i);
                    nb = it->next();
                    while (!it->end())
                    {
                        cf(nb);
                        nb = it->next();
                    }
                }
            },
            parallel);
    }
}

}; }; // end namespace freud::locality

#endif // NEIGHBOR_COMPUTE_FUNCTIONAL_H
