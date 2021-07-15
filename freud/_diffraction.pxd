# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from libcpp cimport bool
from libcpp.vector cimport vector

cimport freud._box
cimport freud._locality
cimport freud.util
from freud._locality cimport BondHistogramCompute
from freud.util cimport vec3

ctypedef unsigned int uint

cdef extern from "StaticStructureFactorDebye.h" namespace "freud::diffraction":
    cdef cppclass StaticStructureFactorDebye:
        StaticStructureFactorDebye(uint, float, float) except +
        void accumulate(const freud._locality.NeighborQuery*,
                        const vec3[float]*,
                        unsigned int) except +
        void reset()
        const freud.util.ManagedArray[float] &getStructureFactor()
        const vector[float] getBinEdges() const
        const vector[float] getBinCenters() const
        float getMinValidK() const
