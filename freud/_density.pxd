# Copyright (c) 2010-2019 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud.util cimport vec3
from libcpp.memory cimport shared_ptr
cimport freud._box
cimport freud._locality

cdef extern from "CorrelationFunction.h" namespace "freud::density":
    cdef cppclass CorrelationFunction[T]:
        CorrelationFunction(float, float) except +
        const freud._box.Box & getBox() const
        void reset()
        void accumulate(const freud._locality.NeighborQuery*, const T*,
                        const vec3[float]*,
                        const T*,
                        unsigned int, const freud._locality.NeighborList*,
                        freud._locality.QueryArgs,) except +
        shared_ptr[T] getRDF()
        shared_ptr[unsigned int] getCounts()
        shared_ptr[float] getR()
        unsigned int getNBins() const

cdef extern from "GaussianDensity.h" namespace "freud::density":
    cdef cppclass GaussianDensity:
        GaussianDensity(vec3[unsigned int], float, float) except +
        const freud._box.Box & getBox() const
        void reset()
        void compute(
            const freud._box.Box &,
            const vec3[float]*,
            unsigned int) except +
        shared_ptr[float] getDensity()
        vec3[unsigned int] getWidth()
        float getSigma()

cdef extern from "LocalDensity.h" namespace "freud::density":
    cdef cppclass LocalDensity:
        LocalDensity(float, float, float)
        const freud._box.Box & getBox() const
        void compute(
            const freud._locality.NeighborQuery*,
            const vec3[float]*,
            unsigned int, const freud._locality.NeighborList *,
            freud._locality.QueryArgs) except +
        unsigned int getNPoints()
        shared_ptr[float] getDensity()
        shared_ptr[float] getNumNeighbors()

cdef extern from "RDF.h" namespace "freud::density":
    cdef cppclass RDF:
        RDF(float, float, float) except +
        const freud._box.Box & getBox() const
        void reset()
        void accumulate(const freud._locality.NeighborQuery*,
                        const vec3[float]*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
        shared_ptr[float] getRDF()
        shared_ptr[float] getR()
        shared_ptr[float] getNr()
        unsigned int getNBins()
