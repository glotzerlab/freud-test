# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cimport freud._box
cimport freud.util
from freud.util cimport vec3


cdef extern from "NeighborBond.h" namespace "freud::locality":
    cdef cppclass NeighborBond:
        unsigned int query_point_idx
        unsigned int point_idx
        float distance
        float weight
        bool operator==(NeighborBond)
        bool operator!=(NeighborBond)
        bool operator<(NeighborBond)

cdef extern from "NeighborQuery.h" namespace "freud::locality":

    ctypedef enum QueryType "freud::locality::QueryType":
        none "freud::locality::QueryType::none"
        ball "freud::locality::QueryType::ball"
        nearest "freud::locality::QueryType::nearest"

    cdef cppclass QueryArgs:
        QueryType mode
        int num_neighbors
        float r_max
        float r_min
        float r_guess
        float scale
        bool exclude_ii

    cdef cppclass NeighborQuery:
        NeighborQuery() except +
        NeighborQuery(const freud._box.Box &,
                      const vec3[float]*,
                      unsigned int) except +
        shared_ptr[NeighborQueryIterator] query(
            const vec3[float]*, unsigned int, QueryArgs) except +
        const freud._box.Box & getBox() const
        const vec3[float]* getPoints const
        const unsigned int getNPoints const
        const vec3[float] operator[](unsigned int) const

    NeighborBond ITERATOR_TERMINATOR \
        "freud::locality::ITERATOR_TERMINATOR"

    cdef cppclass NeighborQueryIterator:
        NeighborQueryIterator()
        NeighborQueryIterator(NeighborQuery*, vec3[float]*, unsigned int)
        bool end()
        NeighborBond next()
        NeighborList *toNeighborList(bool)

cdef extern from "RawPoints.h" namespace "freud::locality":

    cdef cppclass RawPoints(NeighborQuery):
        RawPoints() except +
        RawPoints(const freud._box.Box,
                  const vec3[float]*,
                  unsigned int) except +

cdef extern from "NeighborList.h" namespace "freud::locality":
    cdef cppclass NeighborList:
        NeighborList()
        NeighborList(unsigned int)
        NeighborList(unsigned int, const unsigned int*, unsigned int,
                     const unsigned int*, unsigned int, const vec3[float]*,
                     const float*) except +

        freud.util.ManagedArray[unsigned int] &getNeighbors()
        freud.util.ManagedArray[float] &getDistances()
        freud.util.ManagedArray[float] &getWeights()
        freud.util.ManagedArray[vec3[float]] &getVectors()
        freud.util.ManagedArray[float] &getSegments()
        freud.util.ManagedArray[float] &getCounts()

        unsigned int getNumBonds() const
        unsigned int getNumPoints() const
        unsigned int getNumQueryPoints() const
        void setNumBonds(unsigned int, unsigned int, unsigned int)
        unsigned int filter[Iterator](const Iterator) except +
        unsigned int filter_r(float, float) except +

        unsigned int find_first_index(unsigned int)

        void resize(unsigned int)
        void copy(const NeighborList &)
        void validate(unsigned int, unsigned int) except +

cdef extern from "LinkCell.h" namespace "freud::locality":
    cdef cppclass LinkCell(NeighborQuery):
        LinkCell() except +
        LinkCell(const freud._box.Box &,
                 const vec3[float]*,
                 unsigned int,
                 float) except +
        float getCellWidth() const

cdef extern from "AABBQuery.h" namespace "freud::locality":
    cdef cppclass AABBQuery(NeighborQuery):
        AABBQuery() except +
        AABBQuery(const freud._box.Box,
                  const vec3[float]*,
                  unsigned int) except +

cdef extern from "BondHistogramCompute.h" namespace "freud::locality":
    cdef cppclass BondHistogramCompute:
        BondHistogramCompute()

        const freud._box.Box & getBox() const
        void reset()
        const freud.util.ManagedArray[unsigned int] &getBinCounts()
        vector[vector[float]] getBinEdges() const
        vector[vector[float]] getBinCenters() const
        vector[pair[float, float]] getBounds() const
        vector[size_t] getAxisSizes() const

cdef extern from "PeriodicBuffer.h" namespace "freud::locality":
    cdef cppclass PeriodicBuffer:
        PeriodicBuffer()
        const freud._box.Box & getBox() const
        const freud._box.Box & getBufferBox() const
        void compute(
            const NeighborQuery*,
            const vec3[float],
            const bool) except +
        vector[vec3[float]] getBufferPoints() const
        vector[uint] getBufferIds() const

cdef extern from "Voronoi.h" namespace "freud::locality":
    cdef cppclass Voronoi:
        Voronoi()
        void compute(const NeighborQuery*) nogil except +
        vector[vector[vec3[double]]] getPolytopes() const
        const freud.util.ManagedArray[double] &getVolumes() const
        shared_ptr[NeighborList] getNeighborList() const
