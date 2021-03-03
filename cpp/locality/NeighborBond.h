#ifndef NEIGHBOR_BOND_H
#define NEIGHBOR_BOND_H

namespace freud { namespace locality {

//! Simple data structure encoding neighboring points.
/*! The primary purpose of this class is to provide a more meaningful struct
 *  than a simple std::pair, which is hard to interpret. Additionally, this
 *  class defines the less than operator according to distance, making it
 *  possible to sort.
 */
struct NeighborBond
{
    // For now, id = query_point_idx and ref_id = point_idx (into the NeighborQuery).
    constexpr NeighborBond() = default;

    constexpr NeighborBond(unsigned int query_point_idx, unsigned int point_idx, float d = 0, float w = 1)
        : query_point_idx(query_point_idx), point_idx(point_idx), distance(d), weight(w)
    {}

    //! Equality checks both query_point_idx and distance.
    bool operator==(const NeighborBond& other) const
    {
        return (query_point_idx == other.query_point_idx) && (point_idx == other.point_idx)
            && (distance == other.distance);
    }

    //! Not equals checks inverse of equality.
    bool operator!=(const NeighborBond& other) const
    {
        return !(*this == other);
    }

    //! Default comparator of points is by distance.
    /*! This form of comparison allows easy sorting of nearest neighbors by
     *  distance.
     */
    bool operator<(const NeighborBond& n) const
    {
        return distance < n.distance;
    }

    bool less_id_ref_weight(const NeighborBond& n) const
    {
        if (query_point_idx != n.query_point_idx)
        {
            return query_point_idx < n.query_point_idx;
        }
        if (point_idx != n.point_idx)
        {
            return point_idx < n.point_idx;
        }
        return weight < n.weight;
    }

    bool less_as_tuple(const NeighborBond& n) const
    {
        if (query_point_idx != n.query_point_idx)
        {
            return query_point_idx < n.query_point_idx;
        }
        if (point_idx != n.point_idx)
        {
            return point_idx < n.point_idx;
        }
        if (weight != n.weight)
        {
            return weight < n.weight;
        }
        return distance < n.distance;
    }

    bool less_as_distance(const NeighborBond& n) const
    {
        if (query_point_idx != n.query_point_idx)
        {
            return query_point_idx < n.query_point_idx;
        }
        if (distance != n.distance)
        {
            return distance < n.distance;
        }
        if (point_idx != n.point_idx)
        {
            return point_idx < n.point_idx;
        }
        return weight < n.weight;
    }

    unsigned int query_point_idx {0}; //! The query point index.
    unsigned int point_idx {0};       //! The reference point index.
    float distance {0};               //! The distance between the points.
    float weight {0};                 //! The weight of this bond.
};

}; }; // end namespace freud::locality

#endif // NEIGHBOR_BOND_H
