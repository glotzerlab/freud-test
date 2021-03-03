// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef NEIGHBOR_LIST_H
#define NEIGHBOR_LIST_H

#include <vector>

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborBond.h"
#include "VectorMath.h"

namespace freud { namespace locality {

//! Store a number of near-neighbor bonds from one set of positions
//  ("query points") to another set ("points")
/*! A NeighborList object acts as a source of neighbor information for
    freud's compute methods. Briefly, each bond is associated with a
    query point index, a point index, a distance, and a weight.

    <b>Data structures:</b>

    Query point and point indices are stored in a 2D array m_neighbors of shape
    (n_bonds, 2). The distances and weights arrays are flat per-bond arrays.
 */
class NeighborList
{
public:
    //! Default constructor
    NeighborList();
    //! Create a NeighborList that can hold up to the given number of bonds
    explicit NeighborList(unsigned int num_bonds);
    //! Copy constructor (makes a deep copy)
    NeighborList(const NeighborList& other);
    //! Construct from arrays
    NeighborList(unsigned int num_bonds, const unsigned int* query_point_index, unsigned int num_query_points,
                 const unsigned int* point_index, unsigned int num_points, const float* distances,
                 const float* weights);

    //! Return the number of bonds stored in this NeighborList
    unsigned int getNumBonds() const;
    //! Return the number of query points this NeighborList was built with
    unsigned int getNumQueryPoints() const;
    //! Return the number of points this NeighborList was built with
    unsigned int getNumPoints() const;

    //! Set the number of bonds, query points, and points for this NeighborList object
    void setNumBonds(unsigned int num_bonds, unsigned int num_query_points, unsigned int num_points);
    //! Update the arrays of neighbor counts and segments
    void updateSegmentCounts() const;

    //! Access the neighbors array for reading and writing
    util::ManagedArray<unsigned int>& getNeighbors()
    {
        return m_neighbors;
    }
    //! Access the distances array for reading and writing
    util::ManagedArray<float>& getDistances()
    {
        return m_distances;
    }
    //! Access the weights array for reading and writing
    util::ManagedArray<float>& getWeights()
    {
        return m_weights;
    }
    //! Access the counts array for reading
    util::ManagedArray<unsigned int>& getCounts()
    {
        updateSegmentCounts();
        return m_counts;
    }
    //! Access the segments array for reading
    util::ManagedArray<unsigned int>& getSegments()
    {
        updateSegmentCounts();
        return m_segments;
    }

    //! Access the neighbors array for reading
    const util::ManagedArray<unsigned int>& getNeighbors() const
    {
        return m_neighbors;
    }
    //! Access the distances array for reading
    const util::ManagedArray<float>& getDistances() const
    {
        return m_distances;
    }
    //! Access the weights array for reading
    const util::ManagedArray<float>& getWeights() const
    {
        return m_weights;
    }
    //! Access the counts array for reading
    const util::ManagedArray<unsigned int>& getCounts() const
    {
        updateSegmentCounts();
        return m_counts;
    }
    //! Access the segments array for reading
    const util::ManagedArray<unsigned int>& getSegments() const
    {
        updateSegmentCounts();
        return m_segments;
    }

    //! Remove bonds in this object based on an array of boolean values. The
    //  array must be at least as long as the number of neighbor bonds.
    //  Returns the number of bonds removed.
    template<typename Iterator> unsigned int filter(Iterator begin);
    //! Remove bonds in this object based on minimum and maximum distance
    //  constraints. Returns the number of bonds removed.
    unsigned int filter_r(float r_max, float r_min = 0);

    //! Return the first bond index corresponding to point i
    unsigned int find_first_index(unsigned int i) const;

    //! Resize member arrays to a different size
    void resize(unsigned int num_bonds);

    //! Copy the bonds from another NeighborList object
    void copy(const NeighborList& other);
    //! Throw a runtime_error if num_points and num_query_points do not match
    //  the stored value
    void validate(unsigned int num_query_points, unsigned int num_points) const;

private:
    //! Helper method for bisection search of the neighbor list, used in find_first_index
    unsigned int bisection_search(unsigned int val, unsigned int left, unsigned int right) const;

    //! Number of query points
    unsigned int m_num_query_points;
    //! Number of points
    unsigned int m_num_points;
    //! Neighbor list indices array
    util::ManagedArray<unsigned int> m_neighbors;
    //! Neighbor list per-bond distance array
    util::ManagedArray<float> m_distances;
    //! Neighbor list per-bond weight array
    util::ManagedArray<float> m_weights;

    //! Track whether segments and counts are up to date
    mutable bool m_segments_counts_updated;
    //! Neighbor counts for each query point
    mutable util::ManagedArray<unsigned int> m_counts;
    //! Neighbor segments for each query point
    mutable util::ManagedArray<unsigned int> m_segments;
};

bool compareNeighborBond(const NeighborBond& left, const NeighborBond& right);
bool compareNeighborDistance(const NeighborBond& left, const NeighborBond& right);
bool compareFirstNeighborPairs(const std::vector<NeighborBond>& left, const std::vector<NeighborBond>& right);

}; }; // end namespace freud::locality

#endif // NEIGHBOR_LIST_H
