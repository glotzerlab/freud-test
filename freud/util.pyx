# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

import numpy as np
import freud.box

from functools import wraps

cimport numpy as np

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef class _ManagedArrayContainer:
    """Class responsible for synchronizing ownership between two ManagedArray
    instances.

    The purpose of this container is to minimize memory copies while avoiding
    reference rot. Users obtaining NumPy arrays as outputs from a compute class
    have a reasonable expectation that these arrays will remain valid after the
    compute class recomputes or goes out of scope. To enforce this requirement,
    freud stores all such data members on the C++ side using the ManagedArray
    template class. A ManagedArray shares ownership of its data array through a
    shared pointer, allowing multiple ManagedArrays to share ownership of data.
    Compute classes must use the `prepare` method of the ManagedArray class to
    ensure that they do not overwrite memory spaces shared with other arrays.

    This class creates a copy of a provided ManagedArray to share ownership of
    its data. The resulting _ManagedArrayContainer can be converted to a NumPy
    array and is tied to the lifetime of any such NumPy arrays created from it,
    ensuring that the data remains valid for the lifetime of such objects.

    This class should always be initialized using the static factory
    :meth:`~_ManagedArrayContainer.init` method, which creates the Python copy
    of a ManagedArray provided the instance member of the underlying C++
    compute class.
    """

    def __cinit__(self, arr_type, typenum, element_size):
        # This class should generally be initialized via the factory "init"
        # function, but some logic is included here for ease of use.
        self._element_size = element_size
        self.data_type = arr_type
        self.var_typenum = typenum
        self.thisptr.null_ptr = NULL

    @property
    def shape(self):
        if self.data_type == arr_type_t.UNSIGNED_INT:
            return tuple(self.thisptr.uint_ptr.shape())
        elif self.data_type == arr_type_t.FLOAT:
            return tuple(self.thisptr.float_ptr.shape())
        elif self.data_type == arr_type_t.DOUBLE:
            return tuple(self.thisptr.double_ptr.shape())
        elif self.data_type == arr_type_t.COMPLEX_FLOAT:
            return tuple(self.thisptr.complex_float_ptr.shape())
        elif self.data_type == arr_type_t.COMPLEX_DOUBLE:
            return tuple(self.thisptr.complex_double_ptr.shape())
        elif self.data_type == arr_type_t.BOOL:
            return tuple(self.thisptr.bool_ptr.shape())

    @property
    def element_size(self):
        return self._element_size

    def __dealloc__(self):
        if self.data_type == arr_type_t.UNSIGNED_INT:
            del self.thisptr.uint_ptr
        elif self.data_type == arr_type_t.FLOAT:
            del self.thisptr.float_ptr
        elif self.data_type == arr_type_t.DOUBLE:
            del self.thisptr.double_ptr
        elif self.data_type == arr_type_t.COMPLEX_FLOAT:
            del self.thisptr.complex_float_ptr
        elif self.data_type == arr_type_t.COMPLEX_DOUBLE:
            del self.thisptr.complex_double_ptr
        elif self.data_type == arr_type_t.BOOL:
            del self.thisptr.bool_ptr

    cdef void set_as_base(self, arr):
        """Sets the base of arr to be this object and increases the
        reference count."""
        PyArray_SetBaseObject(arr, self)
        Py_INCREF(self)

    cdef void *get(self):
        """Return a constant raw pointer to the underlying data array."""
        if self.data_type == arr_type_t.UNSIGNED_INT:
            return self.thisptr.uint_ptr.get()
        elif self.data_type == arr_type_t.FLOAT:
            return self.thisptr.float_ptr.get()
        elif self.data_type == arr_type_t.DOUBLE:
            return self.thisptr.double_ptr.get()
        elif self.data_type == arr_type_t.COMPLEX_FLOAT:
            return self.thisptr.complex_float_ptr.get()
        elif self.data_type == arr_type_t.COMPLEX_DOUBLE:
            return self.thisptr.complex_double_ptr.get()
        elif self.data_type == arr_type_t.BOOL:
            return self.thisptr.bool_ptr.get()

    def __array__(self):
        """Convert the underlying data array into a read-only numpy array.

        To simplify the code, we allocate a single linear array and then
        reshape it on return. The reshape is just a view on the arr array
        created below, so it creates a chain reshaped_array->arr->self that
        ensures proper garbage collection.
        """
        cdef np.npy_intp size[1]
        cdef np.ndarray arr
        size[0] = np.prod(self.shape) * self.element_size
        arr = np.PyArray_SimpleNewFromData(
            1, size, self.var_typenum, self.get())

        arr.setflags(write=False)
        self.set_as_base(arr)

        return np.reshape(
            arr, self.shape if self.element_size == 1
            else self.shape + (self.element_size, ))


cdef class _Compute(object):
    R"""Parent class for all compute classes in freud.

    The primary purpose of this class is to prevent access of uncomputed
    values. This is accomplished by maintaining a boolean flag to track whether
    the compute method in a class has been called and decorating class
    properties that rely on compute having been called.

    To use this class, one would write, for example,

    .. code-block:: python
        class Cluster(_Compute):

            def compute(...)
                ...

            @_Compute._computed_property
            def cluster_idx(self):
                return ...

    Attributes:
        _called_compute (bool):
            Flag representing whether the compute method has been called.
    """

    def __cinit__(self):
        self._called_compute = False

    def __getattribute__(self, attr):
        """Compute methods set a flag to indicate that quantities have been
        computed. Compute must be called before plotting."""
        attribute = object.__getattribute__(self, attr)
        if attr == 'compute':
            # Set the attribute *after* computing. This enables
            # self._called_compute to be used in the compute method itself.
            compute = attribute

            @wraps(compute)
            def compute_wrapper(*args, **kwargs):
                return_value = compute(*args, **kwargs)
                self._called_compute = True
                return return_value
            return compute_wrapper
        elif attr == 'plot':
            if not self._called_compute:
                raise AttributeError(
                    "The compute method must be called before calling plot.")
        return attribute

    @staticmethod
    def _computed_property(prop):
        R"""Decorator that makes a class method to be a property with limited access.

        Args:
            prop (callable): The property function.

        Returns:
            Decorator decorating appropriate property method.
        """

        @property
        @wraps(prop)
        def wrapper(self, *args, **kwargs):
            if not self._called_compute:
                raise AttributeError(
                    "Property not computed. Call compute first.")
            return prop(self, *args, **kwargs)
        return wrapper

    def __str__(self):
        return repr(self)


def _convert_array(array, shape=None, dtype=np.float32):
    """Function which takes a given array, checks the dimensions and shape,
    and converts to a supplied dtype.

    Args:
        array (:class:`numpy.ndarray`): Array to check and convert.
        shape: (tuple of int and :code:`None`): Expected shape of the array.
            Only the dimensions that are not :code:`None` are checked.
            (Default value = :code:`None`).
        dtype: :code:`dtype` to convert the array to if :code:`array.dtype`
            is different. If :code:`None`, :code:`dtype` will not be changed
            (Default value = :class:`numpy.float32`).

    Returns:
        :class:`numpy.ndarray`: Array.
    """
    array = np.asarray(array)
    return_arr = np.require(array, dtype=dtype, requirements=['C'])
    if shape is not None:
        if array.ndim != len(shape):
            raise ValueError("array.ndim = {}; expected ndim = {}".format(
                return_arr.ndim, len(shape)))

        for i, s in enumerate(shape):
            if s is not None and return_arr.shape[i] != s:
                shape_str = "(" + ", ".join(str(i) if i is not None
                                            else "..." for i in shape) + ")"
                raise ValueError('array.shape= {}; expected shape = {}'.format(
                    return_arr.shape, shape_str))

    return return_arr


def _convert_box(box, dimensions=None):
    """Function which takes a box-like object and attempts to convert it to
    :class:`freud.box.Box`. Existing :class:`freud.box.Box` objects are
    used directly.

    Args:
        box (box-like object (see :meth:`freud.box.Box.from_box`)): Box to
            check and convert if needed.
        dimensions (int): Number of dimensions the box should be. If not None,
            used to verify the box dimensions (Default value = :code:`None`).

    Returns:
        :class:`freud.box.Box`: freud box.
    """
    if not isinstance(box, freud.box.Box):
        try:
            box = freud.box.Box.from_box(box)
        except ValueError:
            raise

    if dimensions is not None and box.dimensions != dimensions:
        raise ValueError("The box must be {}-dimensional.".format(dimensions))

    return box
