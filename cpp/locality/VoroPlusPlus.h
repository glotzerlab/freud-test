// Copyright (c) 2010-2019 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef VOROPLUSPLUS_H
#define VOROPLUSPLUS_H

#include "Box.h"
#include "VectorMath.h"
#include "NeighborList.h"
#include <voro++/src/voro++.hh>

namespace freud { namespace locality {

struct VoroPlusPlusBond
{
    VoroPlusPlusBond() : index_i(0), index_j(0), weight(0), distance(0) {}

    VoroPlusPlusBond(unsigned int index_i, unsigned int index_j, float w, float d) :
        index_i(index_i), index_j(index_j), weight(w), distance(d) {}

    unsigned int index_i;     //! The point id.
    unsigned int index_j;     //! The reference point id.
    float weight;             //! The weight of this bond.
    float distance;           //! The distance bewteen the points.
};

class VoroPlusPlus
{
public:
    // default constructor
    VoroPlusPlus();

    void compute(const box::Box &box, const vec3<double>* points, unsigned int N);

    NeighborList *getNeighborList()
    {
        return &m_neighbor_list;
    }

    std::vector<std::vector<vec3<double>>> getPolytopes()
    {
        return m_polytopes;
    }

    std::vector<double> getVolumes()
    {
        return m_volumes;
    }

private:
    box::Box m_box;
    NeighborList m_neighbor_list; //!< Stored neighbor list
    std::vector<std::vector<vec3<double>>> m_polytopes; //!< Voronoi polytopes
    std::vector<double> m_volumes; //!< Voronoi cell volumes

};
}; }; // end namespace freud::locality

#endif // VOROPLUSPLUS_H
