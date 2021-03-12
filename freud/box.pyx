# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`~.Box` class defines the geometry of a simulation box. The class
natively supports periodicity by providing the fundamental features for
wrapping vectors outside the box back into it.
"""

import warnings
import numpy as np
import logging

import freud.util

from freud.util cimport vec3
from cpython.object cimport Py_EQ, Py_NE
from libcpp cimport bool as cpp_bool

cimport freud._box
cimport numpy as np

logger = logging.getLogger(__name__)

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Box:
    R"""The freud Box class for simulation boxes.

    This class defines an arbitrary triclinic geometry within which all points are confined.
    By convention, the freud Box is centered at the origin (``[0, 0, 0]``),
    with the extent in each dimension described by the half-open interval ``[-L/2, L/2)``.
    For more information, see the `documentation
    <https://freud.readthedocs.io/en/stable/gettingstarted/tutorial/periodic.html>`_
    on boxes and periodic boundary conditions.

    Also available as ``freud.Box``.

    Args:
        Lx (float, optional):
            The x-dimension length.
        Ly (float, optional):
            The y-dimension length.
        Lz (float, optional):
            The z-dimension length (Default value = 0).
        xy (float, optional):
            The xy tilt factor (Default value = 0).
        xz (float, optional):
            The xz tilt factor (Default value = 0).
        yz (float, optional):
            The yz tilt factor (Default value = 0).
        is2D (bool, optional):
            Whether the box is 2-dimensional. Uses :code:`Lz == 0`
            if :code:`None`. (Default value = :code:`None`)
    """  # noqa: E501

    def __cinit__(self, Lx, Ly, Lz=0, xy=0, xz=0, yz=0, is2D=None):
        if is2D is None:
            is2D = (Lz == 0)
        if is2D:
            if not (Lx and Ly):
                raise ValueError("Lx and Ly must be nonzero for 2D boxes.")
            elif Lz != 0 or xz != 0 or yz != 0:
                warnings.warn(
                    "Specifying z-dimensions in a 2-dimensional box "
                    "has no effect!")
        else:
            if not (Lx and Ly and Lz):
                raise ValueError(
                    "Lx, Ly, and Lz must be nonzero for 3D boxes.")
        self.thisptr = new freud._box.Box(Lx, Ly, Lz, xy, xz, yz, is2D)

    def __dealloc__(self):
        del self.thisptr

    @property
    def L(self):
        """:math:`\\left(3, \\right)` :class:`numpy.ndarray`: Get or set the
        box lengths along x, y, and z."""
        cdef vec3[float] result = self.thisptr.getL()
        return np.asarray([result.x, result.y, result.z])

    @L.setter
    def L(self, value):
        try:
            if len(value) != 3:
                raise ValueError('setL must be called with a scalar or a list '
                                 'of length 3.')
        except TypeError:
            # Will fail if object has no length
            value = (value, value, value)

        if self.is2D and value[2] != 0:
            warnings.warn(
                "Specifying z-dimensions in a 2-dimensional box "
                "has no effect!")
        self.thisptr.setL(value[0], value[1], value[2])

    @property
    def Lx(self):
        """float: Get or set the x-dimension length."""
        return self.thisptr.getLx()

    @Lx.setter
    def Lx(self, value):
        self.L = [value, self.Ly, self.Lz]

    @property
    def Ly(self):
        """float: Get or set the y-dimension length."""
        return self.thisptr.getLy()

    @Ly.setter
    def Ly(self, value):
        self.L = [self.Lx, value, self.Lz]

    @property
    def Lz(self):
        """float: Get or set the z-dimension length."""
        return self.thisptr.getLz()

    @Lz.setter
    def Lz(self, value):
        self.L = [self.Lx, self.Ly, value]

    @property
    def xy(self):
        """float: Get or set the xy tilt factor."""
        return self.thisptr.getTiltFactorXY()

    @xy.setter
    def xy(self, value):
        self.thisptr.setTiltFactorXY(value)

    @property
    def xz(self):
        """float: Get or set the xz tilt factor."""
        return self.thisptr.getTiltFactorXZ()

    @xz.setter
    def xz(self, value):
        self.thisptr.setTiltFactorXZ(value)

    @property
    def yz(self):
        """float: Get or set the yz tilt factor."""
        return self.thisptr.getTiltFactorYZ()

    @yz.setter
    def yz(self, value):
        self.thisptr.setTiltFactorYZ(value)

    @property
    def dimensions(self):
        """int: Get or set the number of dimensions (2 or 3)."""
        return 2 if self.is2D else 3

    @dimensions.setter
    def dimensions(self, value):
        assert value == 2 or value == 3
        self.thisptr.set2D(bool(value == 2))

    @property
    def is2D(self):
        """bool: Whether the box is 2D."""
        return self.thisptr.is2D()

    @property
    def L_inv(self):
        """:math:`\\left(3, \\right)` :class:`numpy.ndarray`: The inverse box
        lengths."""
        cdef vec3[float] result = self.thisptr.getLinv()
        return np.asarray([result.x, result.y, result.z])

    @property
    def volume(self):
        """float: The box volume (area in 2D)."""
        return self.thisptr.getVolume()

    def make_absolute(self, fractional_coordinates):
        R"""Convert fractional coordinates into absolute coordinates.

        Args:
            fractional_coordinates (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Fractional coordinate vector(s), between 0 and 1 within
                parallelepipedal box.

        Returns:
            :math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Absolute coordinate vector(s).
        """  # noqa: E501
        fractions = np.asarray(fractional_coordinates).copy()
        flatten = fractions.ndim == 1
        fractions = np.atleast_2d(fractions)
        fractions = freud.util._convert_array(fractions, shape=(None, 3))

        cdef const float[:, ::1] l_points = fractions
        cdef unsigned int Np = l_points.shape[0]
        self.thisptr.makeAbsolute(<vec3[float]*> &l_points[0, 0], Np)

        return np.squeeze(fractions) if flatten else fractions

    def make_fractional(self, absolute_coordinates):
        R"""Convert absolute coordinates into fractional coordinates.

        Args:
            absolute_coordinates (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Absolute coordinate vector(s).

        Returns:
            :math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Fractional coordinate vector(s).
        """  # noqa: E501
        vecs = np.asarray(absolute_coordinates).copy()
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        vecs = freud.util._convert_array(vecs, shape=(None, 3))

        cdef const float[:, ::1] l_points = vecs
        cdef unsigned int Np = l_points.shape[0]
        self.thisptr.makeFractional(<vec3[float]*> &l_points[0, 0], Np)

        return np.squeeze(vecs) if flatten else vecs

    def get_images(self, vecs):
        R"""Returns the images corresponding to unwrapped vectors.

        Args:
            vecs (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Coordinates of unwrapped vector(s).

        Returns:
            :math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Image index vector(s).
        """  # noqa: E501
        vecs = np.asarray(vecs)
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        vecs = freud.util._convert_array(vecs, shape=(None, 3))

        images = np.zeros(vecs.shape, dtype=np.int32)
        cdef const float[:, ::1] l_points = vecs
        cdef const int[:, ::1] l_result = images
        cdef unsigned int Np = l_points.shape[0]
        self.thisptr.getImages(<vec3[float]*> &l_points[0, 0], Np,
                               <vec3[int]*> &l_result[0, 0])

        return np.squeeze(images) if flatten else images

    def get_box_vector(self, i):
        R"""Get the box vector with index :math:`i`.

        Args:
            i (unsigned int):
                Index (:math:`0 \leq i < d`) of the box vector, where
                :math:`d` is the box dimension (2 or 3).

        Returns:
            :math:`\left(3, \right)` :class:`numpy.ndarray`:
                Box vector with index :math:`i`.
        """
        return self.to_matrix()[:, i]

    @property
    def v1(self):
        """:math:`(3, )` :class:`np.ndarray`: The first box vector
        :math:`(L_x, 0, 0)`."""
        return self.get_box_vector(0)

    @property
    def v2(self):
        """:math:`(3, )` :class:`np.ndarray`: The second box vector
        :math:`(xy*L_y, L_y, 0)`."""
        return self.get_box_vector(1)

    @property
    def v3(self):
        """:math:`(3, )` :class:`np.ndarray`: The third box vector
        :math:`(xz*L_z, yz*L_z, L_z)`."""
        return self.get_box_vector(2)

    def wrap(self, vecs):
        R"""Wrap an array of vectors into the box, using periodic boundaries.

        .. note:: Since the origin of the box is in the center, wrapping is
                  equivalent to applying the minimum image convention to the
                  input vectors.

        Args:
            vecs (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Unwrapped vector(s).

        Returns:
            :math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Vector(s) wrapped into the box.
        """  # noqa: E501
        vecs = np.asarray(vecs)
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        vecs = freud.util._convert_array(vecs, shape=(None, 3)).copy()

        cdef const float[:, ::1] l_points = vecs
        cdef unsigned int Np = l_points.shape[0]
        self.thisptr.wrap(<vec3[float]*> &l_points[0, 0], Np)

        return np.squeeze(vecs) if flatten else vecs

    def unwrap(self, vecs, imgs):
        R"""Unwrap an array of vectors inside the box back into real space,
        using an array of image indices that determine how many times to unwrap
        in each dimension.

        Args:
            vecs (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Vector(s) to be unwrapped.
            imgs (:math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Image indices for vector(s).

        Returns:
            :math:`\left(3, \right)` or :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Unwrapped vector(s).
        """  # noqa: E501
        vecs = np.asarray(vecs)
        flatten = vecs.ndim == 1
        vecs = np.atleast_2d(vecs)
        imgs = np.atleast_2d(imgs)
        if vecs.shape[0] != imgs.shape[0]:
            # Broadcasts (1, 3) to (N, 3) for both arrays
            vecs, imgs = np.broadcast_arrays(vecs, imgs)
        vecs = freud.util._convert_array(vecs, shape=(None, 3)).copy()
        imgs = freud.util._convert_array(imgs, shape=vecs.shape,
                                         dtype=np.int32)

        cdef const float[:, ::1] l_points = vecs
        cdef const int[:, ::1] l_imgs = imgs
        cdef unsigned int Np = l_points.shape[0]
        self.thisptr.unwrap(<vec3[float]*> &l_points[0, 0],
                            <vec3[int]*> &l_imgs[0, 0], Np)

        return np.squeeze(vecs) if flatten else vecs

    def center_of_mass(self, vecs, masses=None):
        R"""Compute center of mass of an array of vectors, using periodic boundaries.

        This calculation accounts for periodic images. `This Wikipedia page
        <https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions>`_
        describes the mathematics of this method.

        Example::

            >>> import freud
            >>> import numpy as np
            >>> box = freud.Box.cube(10)
            >>> points = [[-1, -1, 0], [-1, 1, 0], [2, 0, 0]]
            >>> np.mean(points, axis=0)  # Does not account for periodic images
            array([0., 0., 0.])
            >>> box.center_of_mass(points)  # Accounts for periodic images
            array([-0.1845932,  0.       ,  0.       ])

        Args:
            vecs (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Vectors used to find center of mass.
            masses (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Masses corresponding to each vector, defaulting to 1 if not
                provided or :code:`None` (Default value = :code:`None`).

        Returns:
            :math:`\left(3\right)` :class:`numpy.ndarray`:
                Center of mass.
        """  # noqa: E501
        vecs = freud.util._convert_array(vecs, shape=(None, 3))
        cdef const float[:, ::1] l_points = vecs

        cdef float* l_masses_ptr = NULL
        cdef float[::1] l_masses
        if masses is not None:
            l_masses = freud.util._convert_array(masses, shape=(len(vecs), ))
            l_masses_ptr = &l_masses[0]

        cdef size_t Np = l_points.shape[0]
        cdef vec3[float] result = self.thisptr.centerOfMass(
            <vec3[float]*> &l_points[0, 0], Np, l_masses_ptr)
        return np.asarray([result.x, result.y, result.z])

    def center(self, vecs, masses=None):
        R"""Subtract center of mass from an array of vectors, using periodic boundaries.

        This calculation accounts for periodic images. `This Wikipedia page
        <https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions>`_
        describes the mathematics of this method.

        Example::

            >>> import freud
            >>> box = freud.Box.cube(10)
            >>> points = [[-1, -1, 0], [-1, 1, 0], [2, 0, 0]]
            >>> box.center(points)
            array([[-0.8154068, -1.0000002,  0.       ],
                   [-0.8154068,  1.       ,  0.       ],
                   [ 2.1845937,  0.       ,  0.       ]], dtype=float32)

        Args:
            vecs (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Vectors to center.
            masses (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Masses corresponding to each vector, defaulting to 1 if not
                provided or :code:`None` (Default value = :code:`None`).

        Returns:
            :math:`\left(N, 3\right)` :class:`numpy.ndarray`:
                Vectors with center of mass subtracted.
        """  # noqa: E501
        vecs = freud.util._convert_array(vecs, shape=(None, 3)).copy()
        cdef const float[:, ::1] l_points = vecs

        cdef float* l_masses_ptr = NULL
        cdef float[::1] l_masses
        if masses is not None:
            l_masses = freud.util._convert_array(masses, shape=(len(vecs), ))
            l_masses_ptr = &l_masses[0]

        cdef size_t Np = l_points.shape[0]
        self.thisptr.center(<vec3[float]*> &l_points[0, 0], Np, l_masses_ptr)
        return vecs

    def compute_distances(self, query_points, points):
        R"""Calculate distances between two sets of points, using periodic boundaries.

        Distances are calculated row-wise, i.e. ``distances[i]`` is the
        distance from ``query_points[i]`` to ``points[i]``.

        Args:
            query_points (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Array of query points.
            points (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Array of points.

        Returns:
            :math:`\left(N, \right)` :class:`numpy.ndarray`:
                Array of distances between query points and points.
        """   # noqa: E501

        query_points = freud.util._convert_array(
            np.atleast_2d(query_points), shape=(None, 3))
        points = freud.util._convert_array(
            np.atleast_2d(points), shape=(None, 3))

        cdef:
            const float[:, ::1] l_query_points = query_points
            const float[:, ::1] l_points = points
            size_t n_query_points = query_points.shape[0]
            size_t n_points = points.shape[0]
            float[::1] distances = np.empty(
                n_query_points, dtype=np.float32)

        self.thisptr.computeDistances(
            <vec3[float]*> &l_query_points[0, 0], n_query_points,
            <vec3[float]*> &l_points[0, 0], n_points,
            <float *> &distances[0])
        return np.asarray(distances)

    def compute_all_distances(self, query_points, points):
        R"""Calculate distances between all pairs of query points and points.

        Distances are calculated pairwise, i.e. ``distances[i, j]`` is the
        distance from ``query_points[i]`` to ``points[j]``.

        Args:
            query_points (:math:`\left(N_{query\_points}, 3 \right)` :class:`numpy.ndarray`):
                Array of query points.
            points (:math:`\left(N_{points}, 3 \right)` :class:`numpy.ndarray`):
                Array of points with same length as ``query_points``.

        Returns:
            :math:`\left(N_{query\_points}, N_{points}, \right)` :class:`numpy.ndarray`:
                Array of distances between query points and points.
        """  # noqa: E501
        query_points = freud.util._convert_array(
            np.atleast_2d(query_points), shape=(None, 3))
        points = freud.util._convert_array(
            np.atleast_2d(points), shape=(None, 3))

        cdef:
            const float[:, ::1] l_query_points = query_points
            const float[:, ::1] l_points = points
            size_t n_query_points = query_points.shape[0]
            size_t n_points = points.shape[0]
            float[:, ::1] distances = np.empty(
                [n_query_points, n_points], dtype=np.float32)

        self.thisptr.computeAllDistances(
            <vec3[float]*> &l_query_points[0, 0], n_query_points,
            <vec3[float]*> &l_points[0, 0], n_points,
            <float *> &distances[0, 0])

        return np.asarray(distances)

    def contains(self, points):
        R"""Returns boolean array (mask) corresponding to point membership in a box.

        This calculation computes particle membership based on conventions defined by :class:`Box`, ignoring periodicity.
        This means that in a cubic (3D) box with dimensions ``L``, particles would be considered inside the box if their coordinates are between
        ``[-L/2, L/2]``.
        Particles laying at a coordinate such as ``[0, L, 0]`` would be considered outside the box.
        More information about coordinate conventions can be found `here
        <https://freud.readthedocs.io/en/latest/gettingstarted/examples/module_intros/box.Box.html?highlight=origin#Using-boxes>`__
        and `here <https://freud.readthedocs.io/en/latest/gettingstarted/tutorial/periodic.html?highlight=origin#periodic-boundary-conditions>`__.

        Example::

            >>> import freud
            >>> box = freud.Box.cube(10)
            >>> points = [[-4, 0, 0], [10, 0, 0], [0, -7, 0]]
            >>> box.contains(points)
            array([ True, False, False])

        Args:
            points (:math:`\left(N, 3\right)` :class:`numpy.ndarray`):
                Array of points.

        Returns:
            :math:`\left(N, \right)` :class:`numpy.ndarray`:
                Array of booleans, where `True` corresponds to points within the box,
                and `False` corresponds to points outside the box.
        """  # noqa: E501

        points = freud.util._convert_array(
            np.atleast_2d(points), shape=(None, 3))

        cdef:
            const float[:, ::1] l_points = points
            size_t n_all_points = points.shape[0]

        contains_mask = freud.util._convert_array(
            np.ones(n_all_points), dtype=bool)
        cdef cpp_bool[::1] l_contains_mask = contains_mask

        self.thisptr.contains(
            <vec3[float]*> &l_points[0, 0], n_all_points,
            <cpp_bool*> &l_contains_mask[0])

        return np.array(l_contains_mask).astype(bool)

    @property
    def periodic(self):
        """:math:`\\left(3, \\right)` :class:`numpy.ndarray`: Get or set the
        periodicity of the box in each dimension."""
        periodic = self.thisptr.getPeriodic()
        return np.asarray([periodic.x, periodic.y, periodic.z])

    @periodic.setter
    def periodic(self, periodic):
        # Allow passing a single value
        try:
            self.thisptr.setPeriodic(periodic[0], periodic[1], periodic[2])
        except TypeError:
            # Allow single value to be passed for all directions
            self.thisptr.setPeriodic(periodic, periodic, periodic)

    @property
    def periodic_x(self):
        """bool: Get or set the periodicity of the box in x."""
        return self.thisptr.getPeriodicX()

    @periodic_x.setter
    def periodic_x(self, periodic):
        self.thisptr.setPeriodicX(periodic)

    @property
    def periodic_y(self):
        """bool: Get or set the periodicity of the box in y."""
        return self.thisptr.getPeriodicY()

    @periodic_y.setter
    def periodic_y(self, periodic):
        self.thisptr.setPeriodicY(periodic)

    @property
    def periodic_z(self):
        """bool: Get or set the periodicity of the box in z."""
        return self.thisptr.getPeriodicZ()

    @periodic_z.setter
    def periodic_z(self, periodic):
        self.thisptr.setPeriodicZ(periodic)

    def to_dict(self):
        R"""Return box as dictionary.

        Example::

            >>> box = freud.box.Box.cube(L=10)
            >>> box.to_dict()
            {'Lx': 10.0, 'Ly': 10.0, 'Lz': 10.0,
             'xy': 0.0, 'xz': 0.0, 'yz': 0.0, 'dimensions': 3}

        Returns:
          dict: Box parameters
        """
        return {
            'Lx': self.Lx,
            'Ly': self.Ly,
            'Lz': self.Lz,
            'xy': self.xy,
            'xz': self.xz,
            'yz': self.yz,
            'dimensions': self.dimensions}

    def to_matrix(self):
        R"""Returns the box matrix (3x3).

        Example::

            >>> box = freud.box.Box.cube(L=10)
            >>> box.to_matrix()
            array([[10.,  0.,  0.],
                   [ 0., 10.,  0.],
                   [ 0.,  0., 10.]])

        Returns:
            :math:`\left(3, 3\right)` :class:`numpy.ndarray`: Box matrix
        """
        return np.asarray([[self.Lx, self.xy * self.Ly, self.xz * self.Lz],
                           [0, self.Ly, self.yz * self.Lz],
                           [0, 0, self.Lz]])

    def __repr__(self):
        return ("freud.box.{cls}(Lx={Lx}, Ly={Ly}, Lz={Lz}, "
                "xy={xy}, xz={xz}, yz={yz}, "
                "is2D={is2D})").format(cls=type(self).__name__,
                                       Lx=self.Lx,
                                       Ly=self.Ly,
                                       Lz=self.Lz,
                                       xy=self.xy,
                                       xz=self.xz,
                                       yz=self.yz,
                                       is2D=self.is2D)

    def __str__(self):
        return repr(self)

    def _eq(self, other):
        return self.to_dict() == other.to_dict()

    def __richcmp__(self, other, int op):
        R"""Implement all comparisons for Cython extension classes"""
        if op == Py_EQ:
            return self._eq(other)
        if op == Py_NE:
            return not self._eq(other)
        else:
            raise NotImplementedError("This comparison is not implemented")

    def __mul__(arg1, arg2):
        # Note Cython treats __mul__ and __rmul__ as one operation, so
        # type checks are necessary.
        if isinstance(arg1, freud.box.Box):
            self = arg1
            scale = arg2
        else:
            scale = arg1
            self = arg2
        if scale > 0:
            return self.__class__(Lx=self.Lx*scale,
                                  Ly=self.Ly*scale,
                                  Lz=self.Lz*scale,
                                  xy=self.xy, xz=self.xz, yz=self.yz,
                                  is2D=self.is2D)
        else:
            raise ValueError("Box can only be multiplied by positive values.")

    def plot(self, title=None, ax=None, image=[0, 0, 0], *args, **kwargs):
        """Plot a :class:`~.box.Box` object.

        Args:
            title (str):
                Title of the graph. (Default value = :code:`None`).
            ax (:class:`matplotlib.axes.Axes`):
                Axes object to plot. If :code:`None`, make a new axes and
                figure object. If plotting a 3D box, the axes must be 3D.
                (Default value = :code:`None`).
            image (list):
                The periodic image location at which to draw the box (Default
                value = :code:`[0, 0, 0]`).
            *args:
                Passed on to :meth:`mpl_toolkits.mplot3d.Axes3D.plot` or
                :meth:`matplotlib.axes.Axes.plot`.
            **kwargs:
                Passed on to :meth:`mpl_toolkits.mplot3d.Axes3D.plot` or
                :meth:`matplotlib.axes.Axes.plot`.
        """
        import freud.plot
        return freud.plot.box_plot(self, title=title, ax=ax, image=image,
                                   *args, **kwargs)

    @classmethod
    def from_box(cls, box, dimensions=None):
        R"""Initialize a Box instance from a box-like object.

        Args:
            box:
                A box-like object
            dimensions (int):
                Dimensionality of the box (Default value = None)

        .. note:: Objects that can be converted to freud boxes include
                  lists like :code:`[Lx, Ly, Lz, xy, xz, yz]`,
                  dictionaries with keys
                  :code:`'Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz', 'dimensions'`,
                  objects with attributes
                  :code:`Lx, Ly, Lz, xy, xz, yz, dimensions`,
                  3x3 matrices (see :meth:`~.from_matrix`),
                  or existing :class:`freud.box.Box` objects.

                  If any of :code:`Lz, xy, xz, yz` are not provided, they will
                  be set to 0.

                  If all values are provided, a triclinic box will be
                  constructed. If only :code:`Lx, Ly, Lz` are provided, an
                  orthorhombic box will be constructed. If only :code:`Lx, Ly`
                  are provided, a rectangular (2D) box will be constructed.

                  If the optional :code:`dimensions` argument is given, this
                  will be used as the box dimensionality. Otherwise, the box
                  dimensionality will be detected from the :code:`dimensions`
                  of the provided box. If no dimensions can be detected, the
                  box will be 2D if :code:`Lz == 0`, and 3D otherwise.

        Returns:
            :class:`freud.box.Box`: The resulting box object.
        """
        if np.asarray(box).shape == (3, 3):
            # Handles 3x3 matrices
            return cls.from_matrix(box, dimensions=dimensions)
        try:
            # Handles freud.box.Box and objects with attributes
            Lx = box.Lx
            Ly = box.Ly
            Lz = getattr(box, 'Lz', 0)
            xy = getattr(box, 'xy', 0)
            xz = getattr(box, 'xz', 0)
            yz = getattr(box, 'yz', 0)
            if dimensions is None:
                dimensions = getattr(box, 'dimensions', None)
            elif dimensions != getattr(box, 'dimensions', dimensions):
                raise ValueError(
                    "The provided dimensions argument conflicts with the "
                    "dimensions attribute of the provided box object.")
        except AttributeError:
            try:
                # Handle dictionary-like
                Lx = box['Lx']
                Ly = box['Ly']
                Lz = box.get('Lz', 0)
                xy = box.get('xy', 0)
                xz = box.get('xz', 0)
                yz = box.get('yz', 0)
                if dimensions is None:
                    dimensions = box.get('dimensions', None)
                else:
                    if dimensions != box.get('dimensions', dimensions):
                        raise ValueError(
                            "The provided dimensions argument conflicts with "
                            "the dimensions attribute of the provided box "
                            "object.")
            except (IndexError, KeyError, TypeError):
                if not len(box) in [2, 3, 6]:
                    raise ValueError(
                        "List-like objects must have length 2, 3, or 6 to be "
                        "converted to freud.box.Box.")
                # Handle list-like
                Lx = box[0]
                Ly = box[1]
                Lz = box[2] if len(box) > 2 else 0
                xy, xz, yz = box[3:6] if len(box) == 6 else (0, 0, 0)
        except:  # noqa
            logger.debug('Supplied box cannot be converted to type '
                         'freud.box.Box.')
            raise

        # Infer dimensions if not provided.
        if dimensions is None:
            dimensions = 2 if Lz == 0 else 3
        is2D = (dimensions == 2)
        return cls(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz, is2D=is2D)

    @classmethod
    def from_matrix(cls, box_matrix, dimensions=None):
        R"""Initialize a Box instance from a box matrix.

        For more information and the source for this code,
        see: https://hoomd-blue.readthedocs.io/en/stable/box.html

        Args:
            box_matrix (array-like):
                A 3x3 matrix or list of lists
            dimensions (int):
                Number of dimensions (Default value = :code:`None`)
        """
        box_matrix = np.asarray(box_matrix, dtype=np.float32)
        v0 = box_matrix[:, 0]
        v1 = box_matrix[:, 1]
        v2 = box_matrix[:, 2]
        Lx = np.sqrt(np.dot(v0, v0))
        a2x = np.dot(v0, v1) / Lx
        Ly = np.sqrt(np.dot(v1, v1) - a2x * a2x)
        xy = a2x / Ly
        v0xv1 = np.cross(v0, v1)
        v0xv1mag = np.sqrt(np.dot(v0xv1, v0xv1))
        Lz = np.dot(v2, v0xv1) / v0xv1mag
        if Lz != 0:
            a3x = np.dot(v0, v2) / Lx
            xz = a3x / Lz
            yz = (np.dot(v1, v2) - a2x * a3x) / (Ly * Lz)
        else:
            xz = yz = 0
        if dimensions is None:
            dimensions = 2 if Lz == 0 else 3
        is2D = (dimensions == 2)
        return cls(Lx=Lx, Ly=Ly, Lz=Lz,
                   xy=xy, xz=xz, yz=yz, is2D=is2D)

    @classmethod
    def cube(cls, L=None):
        R"""Construct a cubic box with equal lengths.

        Args:
            L (float): The edge length
        """
        # classmethods compiled with cython don't appear to support
        # named access to positional arguments, so we keep this to
        # recover the behavior
        if L is None:
            raise TypeError("cube() missing 1 required positional argument: L")
        return cls(Lx=L, Ly=L, Lz=L, xy=0, xz=0, yz=0, is2D=False)

    @classmethod
    def square(cls, L=None):
        R"""Construct a 2-dimensional (square) box with equal lengths.

        Args:
            L (float): The edge length
        """
        # classmethods compiled with cython don't appear to support
        # named access to positional arguments, so we keep this to
        # recover the behavior
        if L is None:
            raise TypeError("square() missing 1 required "
                            "positional argument: L")
        return cls(Lx=L, Ly=L, Lz=0, xy=0, xz=0, yz=0, is2D=True)


cdef BoxFromCPP(const freud._box.Box & cppbox):
    b = Box(cppbox.getLx(), cppbox.getLy(), cppbox.getLz(),
            cppbox.getTiltFactorXY(), cppbox.getTiltFactorXZ(),
            cppbox.getTiltFactorYZ(), cppbox.is2D())
    b.periodic = [cppbox.getPeriodicX(),
                  cppbox.getPeriodicY(),
                  cppbox.getPeriodicZ()]
    return b
