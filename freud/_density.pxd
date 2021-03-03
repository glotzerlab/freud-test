# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud.util cimport vec3
from freud._locality cimport BondHistogramCompute
from libcpp cimport bool

cimport freud._box
cimport freud._locality
cimport freud.util

ctypedef unsigned int uint

cdef extern from "CorrelationFunction.h" namespace "freud::density":
    cdef cppclass CorrelationFunction[T](BondHistogramCompute):
        CorrelationFunction(float, float) except +
        void accumulate(const freud._locality.NeighborQuery*, const T*,
                        const vec3[float]*,
                        const T*,
                        unsigned int, const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[T] &getCorrelation()

cdef extern from "GaussianDensity.h" namespace "freud::density":
    cdef cppclass GaussianDensity:
        GaussianDensity(vec3[unsigned int], float, float) except +
        const freud._box.Box & getBox() const
        void reset()
        void compute(const freud._locality.NeighborQuery*,
                     const float*) except +
        const freud.util.ManagedArray[float] &getDensity() const
        vec3[unsigned int] getWidth() const
        float getSigma() const
        float getRMax() const

cdef extern from "LocalDensity.h" namespace "freud::density":
    cdef cppclass LocalDensity:
        LocalDensity(float, float)
        const freud._box.Box & getBox() const
        void compute(
            const freud._locality.NeighborQuery*,
            const vec3[float]*,
            unsigned int, const freud._locality.NeighborList *,
            freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getDensity() const
        const freud.util.ManagedArray[float] &getNumNeighbors() const
        float getRMax() const
        float getDiameter() const

cdef extern from "RDF.h" namespace "freud::density":
    cdef cppclass RDF(BondHistogramCompute):
        RDF(float, float, float, bool) except +
        const freud._box.Box & getBox() const
        void accumulate(const freud._locality.NeighborQuery*,
                        const vec3[float]*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
        const freud.util.ManagedArray[float] &getRDF()
        const freud.util.ManagedArray[float] &getNr()

cdef extern from "SphereVoxelization.h" namespace "freud::density":
    cdef cppclass SphereVoxelization:
        SphereVoxelization(vec3[unsigned int], float) except +
        const freud._box.Box & getBox() const
        void reset()
        void compute(const freud._locality.NeighborQuery*) except +
        const freud.util.ManagedArray[unsigned int] &getVoxels() const
        vec3[unsigned int] getWidth() const
        float getRMax() const
