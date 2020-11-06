// Copyright (c) 2010-2020 The Regents of the University of Michigan
// This file is from the freud project, released under the BSD 3-Clause License.

#ifndef VORONOI_H
#define VORONOI_H

#include "Box.h"
#include "ManagedArray.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "VectorMath.h"
#include <voro++/src/voro++.hh>

namespace freud { namespace locality {

class Voronoi
{
public:
    // default constructor
    Voronoi() : m_neighbor_list(std::make_shared<NeighborList>()) {}

    void compute(const freud::locality::NeighborQuery* nq, const double* radii = nullptr);

    std::shared_ptr<NeighborList> getNeighborList() const
    {
        return m_neighbor_list;
    }

    const std::vector<std::vector<vec3<double>>> getPolytopes() const
    {
        return m_polytopes;
    }

    const util::ManagedArray<double>& getVolumes() const
    {
        return m_volumes;
    }

private:
    box::Box m_box;
    std::shared_ptr<NeighborList> m_neighbor_list;      //!< Stored neighbor list
    std::vector<std::vector<vec3<double>>> m_polytopes; //!< Voronoi polytopes
    util::ManagedArray<double> m_volumes;               //!< Voronoi cell volumes
};
}; }; // end namespace freud::locality

#endif // VORONOI_H
