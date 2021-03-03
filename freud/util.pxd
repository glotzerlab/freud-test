# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

# Directly expose vec3 and quat since they're ubiquitous in constructing
# arguments to interface with the C++ implementations of all methods.
import numpy as np

from freud._util cimport vec3, quat, ManagedArray, PyArray_SetBaseObject
from cpython cimport Py_INCREF
from libcpp.complex cimport complex
from cython.operator cimport dereference
from libcpp cimport bool

cimport numpy as np

ctypedef unsigned int uint
ctypedef float complex fcomplex
ctypedef double complex dcomplex

ctypedef enum arr_type_t:
    FLOAT
    DOUBLE
    COMPLEX_FLOAT
    COMPLEX_DOUBLE
    UNSIGNED_INT
    BOOL


ctypedef union arr_ptr_t:
    void *null_ptr
    ManagedArray[float] *float_ptr
    ManagedArray[double] *double_ptr
    ManagedArray[float complex] *complex_float_ptr
    ManagedArray[double complex] *complex_double_ptr
    ManagedArray[uint] *uint_ptr
    ManagedArray[bool] *bool_ptr


cdef class _ManagedArrayContainer:
    cdef uint _element_size
    cdef int var_typenum
    cdef arr_ptr_t thisptr
    cdef arr_type_t data_type

    cdef void set_as_base(self, arr)
    cdef void *get(self)

    @staticmethod
    cdef inline _ManagedArrayContainer init(
            const void *array, arr_type_t arr_type, uint element_size=1):
        cdef _ManagedArrayContainer obj

        if arr_type == arr_type_t.FLOAT:
            obj = _ManagedArrayContainer(arr_type, np.NPY_FLOAT,
                                         element_size)
            obj.thisptr.float_ptr = new ManagedArray[float](
                dereference(<const ManagedArray[float] *>array))
        elif arr_type == arr_type_t.DOUBLE:
            obj = _ManagedArrayContainer(arr_type, np.NPY_DOUBLE,
                                         element_size)
            obj.thisptr.double_ptr = new ManagedArray[double](
                dereference(<const ManagedArray[double] *>array))
        elif arr_type == arr_type_t.COMPLEX_FLOAT:
            obj = _ManagedArrayContainer(arr_type, np.NPY_COMPLEX64,
                                         element_size)
            obj.thisptr.complex_float_ptr = new ManagedArray[fcomplex](
                dereference(<const ManagedArray[fcomplex] *>array))
        elif arr_type == arr_type_t.COMPLEX_DOUBLE:
            obj = _ManagedArrayContainer(arr_type, np.NPY_COMPLEX128,
                                         element_size)
            obj.thisptr.complex_double_ptr = new ManagedArray[dcomplex](
                dereference(<const ManagedArray[dcomplex] *>array))
        elif arr_type == arr_type_t.UNSIGNED_INT:
            obj = _ManagedArrayContainer(arr_type, np.NPY_UINT32,
                                         element_size)
            obj.thisptr.uint_ptr = new ManagedArray[uint](
                dereference(<const ManagedArray[uint] *>array))
        elif arr_type == arr_type_t.BOOL:
            obj = _ManagedArrayContainer(arr_type, np.NPY_BOOL,
                                         element_size)
            obj.thisptr.bool_ptr = new ManagedArray[bool](
                dereference(<const ManagedArray[bool] *>array))

        return obj


cdef inline make_managed_numpy_array(
        const void *array, arr_type_t arr_type, uint element_size=1):
    """Make a _ManagedArrayContainer and return an array pointing to its
    data."""
    return np.asarray(
        _ManagedArrayContainer.init(array, arr_type, element_size))


cdef class _Compute:
    cdef public bool _called_compute
