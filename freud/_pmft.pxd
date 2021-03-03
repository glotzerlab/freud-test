# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from freud.util cimport vec3, quat
from freud._locality cimport BondHistogramCompute

cimport freud._locality
cimport freud.util

cdef extern from "PMFT.h" namespace "freud::pmft":
    cdef cppclass PMFT(BondHistogramCompute):
        PMFT() except +
        const freud.util.ManagedArray[float] &getPCF()

cdef extern from "PMFTR12.h" namespace "freud::pmft":
    cdef cppclass PMFTR12(PMFT):
        PMFTR12(float, unsigned int, unsigned int, unsigned int) except +

        void accumulate(const freud._locality.NeighborQuery*,
                        const float*,
                        const vec3[float]*,
                        const float*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +

cdef extern from "PMFTXYT.h" namespace "freud::pmft":
    cdef cppclass PMFTXYT(PMFT):
        PMFTXYT(float, float,
                unsigned int, unsigned int, unsigned int) except +

        void accumulate(const freud._locality.NeighborQuery*,
                        const float*,
                        const vec3[float]*,
                        const float*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +

cdef extern from "PMFTXY.h" namespace "freud::pmft":
    cdef cppclass PMFTXY(PMFT):
        PMFTXY(float, float, unsigned int, unsigned int) except +

        void accumulate(const freud._locality.NeighborQuery*,
                        const float*,
                        const vec3[float]*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +

cdef extern from "PMFTXYZ.h" namespace "freud::pmft":
    cdef cppclass PMFTXYZ(PMFT):
        PMFTXYZ(float, float, float, unsigned int, unsigned int,
                unsigned int, vec3[float]) except +

        void accumulate(const freud._locality.NeighborQuery*,
                        const quat[float]*,
                        const vec3[float]*,
                        unsigned int,
                        const quat[float]*,
                        unsigned int,
                        const freud._locality.NeighborList*,
                        freud._locality.QueryArgs) except +
