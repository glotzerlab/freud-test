# Copyright (c) 2010-2020 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

R"""
The :class:`freud.order` module contains functions which compute order
parameters for the whole system or individual particles. Order parameters take
bond order data and interpret it in some way to quantify the degree of order in
a system using a scalar value. This is often done through computing spherical
harmonics of the bond order diagram, which are the spherical analogue of
Fourier Transforms.
"""

import numpy as np
import time
import freud.locality
import logging
import warnings

from freud.util cimport _Compute, vec3, quat
from freud.errors import FreudDeprecationWarning
from freud.locality cimport _PairCompute
from cython.operator cimport dereference

cimport freud._order
cimport freud.locality
cimport freud.util
cimport numpy as np

logger = logging.getLogger(__name__)

# numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef class Cubatic(_Compute):
    R"""Compute the cubatic order parameter :cite:`Haji_Akbari_2015` for a system of
    particles using simulated annealing instead of Newton-Raphson root finding.

    Args:
        t_initial (float):
            Starting temperature.
        t_final (float):
            Final temperature.
        scale (float):
            Scaling factor to reduce temperature.
        n_replicates (unsigned int, optional):
            Number of replicate simulated annealing runs.
            (Default value = :code:`1`).
        seed (unsigned int, optional):
            Random seed to use in calculations. If :code:`None`, system time is used.
            (Default value = :code:`None`).
    """  # noqa: E501
    cdef freud._order.Cubatic * thisptr

    def __cinit__(self, t_initial, t_final, scale, n_replicates=1, seed=None):
        if seed is None:
            seed = int(time.time())

        self.thisptr = new freud._order.Cubatic(
            t_initial, t_final, scale, n_replicates, seed)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, orientations):
        R"""Calculates the per-particle and global order parameter.

        Args:
            orientations ((:math:`N_{particles}`, 4) :class:`numpy.ndarray`):
                Orientations as angles to use in computation.
        """
        orientations = freud.util._convert_array(
            orientations, shape=(None, 4))

        cdef const float[:, ::1] l_orientations = orientations
        cdef unsigned int num_particles = l_orientations.shape[0]

        self.thisptr.compute(
            <quat[float]*> &l_orientations[0, 0], num_particles)
        return self

    @property
    def t_initial(self):
        """float: The value of the initial temperature."""
        return self.thisptr.getTInitial()

    @property
    def t_final(self):
        """float: The value of the final temperature."""
        return self.thisptr.getTFinal()

    @property
    def scale(self):
        """float: The scale."""
        return self.thisptr.getScale()

    @property
    def n_replicates(self):
        """unsigned int: Number of replicate simulated annealing runs."""
        return self.thisptr.getNReplicates()

    @property
    def seed(self):
        """unsigned int: Random seed to use in calculations."""
        return self.thisptr.getSeed()

    @_Compute._computed_property
    def order(self):
        """float: Cubatic order parameter of the system."""
        return self.thisptr.getCubaticOrderParameter()

    @_Compute._computed_property
    def orientation(self):
        """:math:`\\left(4 \\right)` :class:`numpy.ndarray`: The quaternion of
        global orientation."""
        cdef quat[float] q = self.thisptr.getCubaticOrientation()
        return np.asarray([q.s, q.v.x, q.v.y, q.v.z], dtype=np.float32)

    @_Compute._computed_property
    def particle_order(self):
        """:math:`\\left(N_{particles} \\right)` :class:`numpy.ndarray`: Order
        parameter."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getParticleOrderParameter(),
            freud.util.arr_type_t.FLOAT)

    @_Compute._computed_property
    def global_tensor(self):
        """:math:`\\left(3, 3, 3, 3 \\right)` :class:`numpy.ndarray`: Rank 4
        tensor corresponding to the global orientation. Computed from all
        orientations."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getGlobalTensor(),
            freud.util.arr_type_t.FLOAT)

    @_Compute._computed_property
    def cubatic_tensor(self):
        """:math:`\\left(3, 3, 3, 3 \\right)` :class:`numpy.ndarray`: Rank 4
        homogeneous tensor representing the optimal system-wide coordinates."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getCubaticTensor(),
            freud.util.arr_type_t.FLOAT)

    def __repr__(self):
        return ("freud.order.{cls}(t_initial={t_initial}, t_final={t_final}, "
                "scale={scale}, n_replicates={n_replicates}, "
                "seed={seed})").format(cls=type(self).__name__,
                                       t_initial=self.t_initial,
                                       t_final=self.t_final,
                                       scale=self.scale,
                                       n_replicates=self.n_replicates,
                                       seed=self.seed)


cdef class Nematic(_Compute):
    R"""Compute the nematic order parameter for a system of particles.

    Args:
        u (:math:`\left(3 \right)` :class:`numpy.ndarray`):
            The nematic director of a single particle in the reference state
            (without any rotation applied).
    """
    cdef freud._order.Nematic *thisptr

    def __cinit__(self, u):
        # run checks
        if len(u) != 3:
            raise ValueError('u needs to be a three-dimensional vector')

        cdef vec3[float] l_u = vec3[float](u[0], u[1], u[2])
        self.thisptr = new freud._order.Nematic(l_u)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, orientations):
        R"""Calculates the per-particle and global order parameter.

        Example::

            >>> orientations = np.array([[1, 0, 0, 0]] * 100)
            >>> director = np.array([1, 1, 0])
            >>> nematic = freud.order.Nematic(director)
            >>> nematic.compute(orientations)
            freud.order.Nematic(u=[...])

        Args:
            orientations (:math:`\left(N_{particles}, 4 \right)` :class:`numpy.ndarray`):
                Orientations to calculate the order parameter.
        """   # noqa: E501
        orientations = freud.util._convert_array(
            orientations, shape=(None, 4))

        cdef const float[:, ::1] l_orientations = orientations
        cdef unsigned int num_particles = l_orientations.shape[0]

        self.thisptr.compute(<quat[float]*> &l_orientations[0, 0],
                             num_particles)
        return self

    @_Compute._computed_property
    def order(self):
        """float: Nematic order parameter of the system."""
        return self.thisptr.getNematicOrderParameter()

    @_Compute._computed_property
    def director(self):
        """:math:`\\left(3 \\right)` :class:`numpy.ndarray`: The average
        nematic director."""
        cdef vec3[float] n = self.thisptr.getNematicDirector()
        return np.asarray([n.x, n.y, n.z], dtype=np.float32)

    @_Compute._computed_property
    def particle_tensor(self):
        """:math:`\\left(N_{particles}, 3, 3 \\right)` :class:`numpy.ndarray`:
            One 3x3 matrix per-particle corresponding to each individual
            particle orientation."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getParticleTensor(),
            freud.util.arr_type_t.FLOAT)

    @_Compute._computed_property
    def nematic_tensor(self):
        """:math:`\\left(3, 3 \\right)` :class:`numpy.ndarray`: 3x3 matrix
        corresponding to the average particle orientation."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getNematicTensor(),
            freud.util.arr_type_t.FLOAT)

    @property
    def u(self):
        """:math:`\\left(3 \\right)` :class:`numpy.ndarray`: The normalized
        reference director (the normalized vector provided on construction)."""
        cdef vec3[float] u = self.thisptr.getU()
        return np.asarray([u.x, u.y, u.z], dtype=np.float32)

    def __repr__(self):
        return "freud.order.{cls}(u={u})".format(cls=type(self).__name__,
                                                 u=self.u.tolist())


cdef class Hexatic(_PairCompute):
    R"""Calculates the :math:`k`-atic order parameter for 2D systems.

    The :math:`k`-atic order parameter (called the hexatic order parameter for
    :math:`k = 6`) is analogous to Steinhardt order parameters, and is used to
    measure order in the bonds of 2D systems.

    The :math:`k`-atic order parameter for a particle :math:`i` and its
    :math:`n` neighbors :math:`j` is given by:

    :math:`\psi_k \left( i \right) = \frac{1}{n}
    \sum_j^n e^{i k \phi_{ij}}`

    The parameter :math:`k` governs the symmetry of the order parameter and
    typically matches the number of neighbors to be found for each particle.
    The quantity :math:`\phi_{ij}` is the angle between the
    vector :math:`r_{ij}` and :math:`\left(1, 0\right)`.

    If the weighted mode is enabled, contributions of each neighbor are
    weighted. Neighbor weights :math:`w_j` default to 1 but are defined for a
    :class:`freud.locality.NeighborList` from :class:`freud.locality.Voronoi`
    or one with user-provided weights. The formula is modified as follows:

    :math:`\psi'_k \left( i \right) = \frac{1}{\sum_j^n w_j}
    \sum_j^n w_j e^{i k \phi_{ij}}`

    The hexatic order parameter as written above is **complex-valued**. The
    **magnitude** of the complex value,
    :code:`np.abs(hex_order.particle_order)`, is frequently what is desired
    when determining the :math:`k`-atic order for each particle. The complex
    phase angle :code:`np.angle(hex_order.particle_order)` indicates the
    orientation of the bonds as an angle measured counterclockwise from the
    vector :math:`\left(1, 0\right)`. The complex valued order parameter is
    not rotationally invariant because of this phase angle, but the magnitude
    *is* rotationally invariant.

    .. note::
        **2D:** :class:`freud.order.Hexatic` is only defined for 2D systems.
        The points must be passed in as :code:`[x, y, 0]`.

    Args:
        k (unsigned int, optional):
            Symmetry of order parameter (Default value = :code:`6`).
        weighted (bool, optional):
            Determines whether to use neighbor weights in the computation of
            spherical harmonics over neighbors. If enabled and used with a
            Voronoi neighbor list, this results in the 2D Minkowski Structure
            Metrics :math:`\psi'_k` :cite:`Mickel2013` (Default value =
            :code:`False`).
    """  # noqa: E501
    cdef freud._order.Hexatic * thisptr

    def __cinit__(self, k=6, weighted=False):
        self.thisptr = new freud._order.Hexatic(k, weighted)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, system, neighbors=None):
        R"""Calculates the hexatic order parameter.

        Example::

            >>> box, points = freud.data.make_random_system(
            ...     box_size=10, num_points=100, is2D=True, seed=0)
            >>> # Compute the hexatic (6-fold) order for the 2D system
            >>> hex_order = freud.order.Hexatic(k=6)
            >>> hex_order.compute(system=(box, points))
            freud.order.Hexatic(...)
            >>> print(hex_order.particle_order)
            [...]

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
        """   # noqa: E501
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(system, neighbors=neighbors)
        self.thisptr.compute(nlist.get_ptr(),
                             nq.get_ptr(), dereference(qargs.thisptr))
        return self

    @property
    def default_query_args(self):
        """The default query arguments are
        :code:`{'mode': 'nearest', 'num_neighbors': self.k}`."""
        return dict(mode="nearest", num_neighbors=self.k)

    @_Compute._computed_property
    def particle_order(self):
        """:math:`\\left(N_{particles} \\right)` :class:`numpy.ndarray`: Order
        parameter."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getOrder(),
            freud.util.arr_type_t.COMPLEX_FLOAT)

    @property
    def k(self):
        """unsigned int: Symmetry of the order parameter."""
        return self.thisptr.getK()

    @property
    def weighted(self):
        """bool: Whether neighbor weights were used in the computation."""
        return self.thisptr.isWeighted()

    def __repr__(self):
        return "freud.order.{cls}(k={k}, weighted={weighted})".format(
            cls=type(self).__name__, k=self.k, weighted=self.weighted)

    def plot(self, ax=None):
        """Plot order parameter distribution.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis
                (Default value = :code:`None`).

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        xlabel = r"$\left|\psi{prime}_{k}\right|$".format(
            prime='\'' if self.weighted else '',
            k=self.k)

        return freud.plot.histogram_plot(
            np.absolute(self.particle_order),
            title="Hexatic Order Parameter " + xlabel,
            xlabel=xlabel,
            ylabel=r"Number of particles",
            ax=ax)

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


cdef class Translational(_PairCompute):
    R"""Compute the translational order parameter for each particle.

    The translational order parameter is used to measure order in the bonds
    of 2D systems. The translational order parameter for a particle :math:`i`
    and its :math:`n` neighbors :math:`j` is given by a sum over the
    neighbors, treating the 2D vectors between each pair of particles as a
    complex number with real part corresponding to the x-component of the
    vector and imaginary part corresponding to the y-component of the vector,
    divided by a normalization constant :math:`k`:

    :math:`\psi\left( i \right) = \frac{1}{k} \sum_j^n x_{ij} + y_{ij} i`

    The translational order parameter as written above is **complex-valued**.

    .. note::
        **2D:** :class:`freud.order.Translational` is only defined for 2D
        systems. The points must be passed in as :code:`[x, y, 0]`.

    .. note::
        This class is slated for deprecation and will be removed in freud 3.0.

    Args:
        k (float, optional):
            Normalization of order parameter (Default value = :code:`6.0`).
    """  # noqa E501
    cdef freud._order.Translational * thisptr

    def __cinit__(self, k=6.0):
        warnings.warn("This class is deprecated and will be removed in "
                      "version 3.0", FreudDeprecationWarning)
        self.thisptr = new freud._order.Translational(k, False)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, system, neighbors=None):
        R"""Calculates the local descriptors.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
        """
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(system, neighbors=neighbors)

        self.thisptr.compute(nlist.get_ptr(),
                             nq.get_ptr(), dereference(qargs.thisptr))
        return self

    @property
    def default_query_args(self):
        """The default query arguments are
        :code:`{'mode': 'nearest', 'num_neighbors': int(self.k)}`."""
        return dict(mode="nearest", num_neighbors=int(self.k))

    @_Compute._computed_property
    def particle_order(self):
        """:math:`\\left(N_{particles} \\right)` :class:`numpy.ndarray`: Order
        parameter."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getOrder(),
            freud.util.arr_type_t.COMPLEX_FLOAT)

    @property
    def k(self):
        """float: Normalization of the order parameter."""
        return self.thisptr.getK()

    def __repr__(self):
        return "freud.order.{cls}(k={k})".format(
            cls=type(self).__name__, k=self.k)


cdef class Steinhardt(_PairCompute):
    R"""Compute the rotationally invariant Steinhardt order parameter
    :math:`q_l` or :math:`w_l` for a set of points :cite:`Steinhardt:1983aa`.

    Implements the local rotationally invariant :math:`q_l` or :math:`w_l`
    order parameter described by Steinhardt. For a particle :math:`i`, we
    calculate the average order parameter by summing the spherical harmonics
    between particle :math:`i` and its neighbors :math:`j` in a local region:

    :math:`\overline{q}_{lm}(i) = \frac{1}{N_b} \displaystyle\sum_{j=1}^{N_b}
    Y_{lm}(\theta(\vec{r}_{ij}), \phi(\vec{r}_{ij}))`.

    For :math:`q_l`, this is then combined in a rotationally invariant fashion
    to remove local orientational order as follows:

    :math:`q_l(i)=\sqrt{\frac{4\pi}{2l+1} \displaystyle\sum_{m=-l}^{l}
    |\overline{q}_{lm}|^2 }`.

    For :math:`w_l`, it is then defined as a weighted average over the
    :math:`\overline{q}_{lm}(i)` values using Wigner 3j symbols (Clebsch-Gordan
    coefficients). The resulting combination is rotationally (i.e. frame)
    invariant.

    The average argument in the constructor provides access to a variant of
    this parameter that performs a average over the first and second shell
    combined :cite:`Lechner_2008`. To compute this parameter, we perform a second
    averaging over the first neighbor shell of the particle to implicitly
    include information about the second neighbor shell. This averaging is
    performed by replacing the value :math:`\overline{q}_{lm}(i)` in the
    original definition by the average value of :math:`\overline{q}_{lm}(k)`
    over all the :math:`k` neighbors of particle :math:`i` as well as itself.

    If the weighted mode is enabled in the constructor, the contributions of
    each neighbor are weighted. Neighbor weights :math:`w_j` default to 1 but
    are defined for a :class:`freud.locality.NeighborList` from
    :class:`freud.locality.Voronoi` or one with user-provided weights. The
    formula is modified as follows:

    :math:`\overline{q}'_{lm}(i) = \frac{1}{\sum_j^n w_j} \displaystyle\sum_{j=1}^{N_b}
    w_j Y_{lm}(\theta(\vec{r}_{ij}), \phi(\vec{r}_{ij}))`.

    :math:`q'_l(i)=\sqrt{\frac{4\pi}{2l+1} \displaystyle\sum_{m=-l}^{l}
    |\overline{q}'_{lm}|^2 }`.

    The :code:`norm` attribute argument provides normalized versions of the
    order parameter, where the normalization is performed by averaging the
    :math:`q_{lm}` values over all particles before computing the order
    parameter of choice.

    .. note::
        The value of per-particle order parameter will be set to NaN for
        particles with no neighbors. We choose this value rather than setting
        the order parameter to 0 because in more complex order parameter
        calculations (such as when computing the :math:`w_l`), it is possible
        to observe a value of 0 for the per-particle order parameter even with
        a finite number of neighbors. If you would like to ignore this
        distinction, you can mask the output order parameter values using
        NumPy: :code:`numpy.nan_to_num(particle_order)`.

    Args:
        l (unsigned int):
            Spherical harmonic quantum number l.
        average (bool, optional):
            Determines whether to calculate the averaged Steinhardt order
            parameter (Default value = :code:`False`).
        wl (bool, optional):
            Determines whether to use the :math:`w_l` version of the Steinhardt
            order parameter (Default value = :code:`False`).
        weighted (bool, optional):
            Determines whether to use neighbor weights in the computation of
            spherical harmonics over neighbors. If enabled and used with a
            Voronoi neighbor list, this results in the 3D Minkowski Structure
            Metrics :math:`q'_l` :cite:`Mickel2013` (Default value =
            :code:`False`).
        wl_normalize (bool, optional):
            Determines whether to normalize the :math:`w_l` version
            of the Steinhardt order parameter (Default value = :code:`False`).
    """  # noqa: E501
    cdef freud._order.Steinhardt * thisptr

    def __cinit__(self, l, average=False, wl=False, weighted=False,
                  wl_normalize=False):
        self.thisptr = new freud._order.Steinhardt(l, average, wl, weighted,
                                                   wl_normalize)

    def __dealloc__(self):
        del self.thisptr

    @property
    def average(self):
        """bool: Whether the averaged Steinhardt order parameter was
        calculated."""
        return self.thisptr.isAverage()

    @property
    def wl(self):
        """bool: Whether the :math:`W_l` version of the Steinhardt order
        parameter was used."""
        return self.thisptr.isWl()

    @property
    def weighted(self):
        """bool: Whether neighbor weights were used in the computation."""
        return self.thisptr.isWeighted()

    @property
    def wl_normalize(self):
        return self.thisptr.isWlNormalized()

    @property
    def l(self):  # noqa: E743
        """unsigned int: Spherical harmonic quantum number l."""
        return self.thisptr.getL()

    @_Compute._computed_property
    def order(self):
        """float: The system wide normalization of the :math:`q_l` or
        :math:`w_l` order parameter."""
        return self.thisptr.getOrder()

    @_Compute._computed_property
    def particle_order(self):
        """:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`: Variant
        of the Steinhardt order parameter for each particle (filled with
        :code:`nan` for particles with no neighbors)."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getParticleOrder(),
            freud.util.arr_type_t.FLOAT)

    @_Compute._computed_property
    def ql(self):
        """:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
        :math:`q_l` Steinhardt order parameter for each particle (filled with
        :code:`nan` for particles with no neighbors). This is always available,
        no matter which options are selected."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getQl(),
            freud.util.arr_type_t.FLOAT)

    def compute(self, system, neighbors=None):
        R"""Compute the order parameter.

        Example::

            >>> box, points = freud.data.make_random_system(10, 100, seed=0)
            >>> ql = freud.order.Steinhardt(l=6)
            >>> ql.compute((box, points), {'r_max':3})
            freud.order.Steinhardt(l=6, ...)

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
        """   # noqa: E501
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(system, neighbors=neighbors)

        self.thisptr.compute(nlist.get_ptr(),
                             nq.get_ptr(),
                             dereference(qargs.thisptr))
        return self

    def __repr__(self):
        return ("freud.order.{cls}(l={l}, average={average}, wl={wl}, "
                "weighted={weighted}, wl_normalize={wl_normalize})").format(
                    cls=type(self).__name__,
                    l=self.l, # noqa: 743
                    average=self.average,
                    wl=self.wl,
                    weighted=self.weighted,
                    wl_normalize=self.wl_normalize)

    def plot(self, ax=None):
        """Plot order parameter distribution.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis
                (Default value = :code:`None`).

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        xlabel = r"${mode_letter}{prime}_{{{sph_l}{average}}}$".format(
            mode_letter='w' if self.wl else 'q',
            prime='\'' if self.weighted else '',
            sph_l=self.l,
            average=',ave' if self.average else '')

        return freud.plot.histogram_plot(
            self.particle_order,
            title="Steinhardt Order Parameter " + xlabel,
            xlabel=xlabel,
            ylabel=r"Number of particles",
            ax=ax)

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


cdef class SolidLiquid(_PairCompute):
    R"""Identifies solid-like clusters using dot products of :math:`q_{lm}`.

    The solid-liquid order parameter :cite:`Wolde:1995aa,Filion_2010` uses a
    Steinhardt-like approach to identify solid-like particles. First, a bond
    parameter :math:`q_l(i, j)` is computed for each neighbor bond.

    If :code:`normalize_q` is true (default), the bond parameter is given by
    :math:`q_l(i, j) = \frac{\sum_{m=-l}^{l} \text{Re}~q_{lm}(i) q_{lm}^*(j)}
    {\sqrt{\sum_{m=-l}^{l} \lvert q_{lm}(i) \rvert^2}
    \sqrt{\sum_{m=-l}^{l} \lvert q_{lm}(j) \rvert^2}}`

    If :code:`normalize_q` is false, then the denominator of the above
    expression is left out.

    Next, the bonds are filtered to keep only "solid-like" bonds with
    :math:`q_l(i, j)` above a cutoff value :math:`q_{threshold}`.

    If a particle has more than :math:`S_{threshold}` solid-like bonds, then
    the particle is considered solid-like. Finally, solid-like particles are
    clustered.

    Args:
        l (unsigned int):
            Spherical harmonic quantum number l.
        q_threshold (float):
            Value of dot product threshold when evaluating
            :math:`q_l(i, j)` to determine if a bond is solid-like. For
            :math:`l=6`, 0.7 is generally good for FCC or BCC structures
            :cite:`Filion_2010`.
        solid_threshold (unsigned int):
            Minimum required number of adjacent solid-like bonds for a particle
            to be considered solid-like for clustering. For :math:`l=6`, 6-8
            is generally good for FCC or BCC structures.
        normalize_q (bool):
            Whether to normalize the dot product (Default value =
            :code:`True`).
    """  # noqa: E501
    cdef freud._order.SolidLiquid * thisptr

    def __cinit__(self, l, q_threshold, solid_threshold, normalize_q=True):
        self.thisptr = new freud._order.SolidLiquid(
            l, q_threshold, solid_threshold, normalize_q)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, system, neighbors=None):
        R"""Compute the order parameter.

        Args:
            system:
                Any object that is a valid argument to
                :class:`freud.locality.NeighborQuery.from_system`.
            neighbors (:class:`freud.locality.NeighborList` or dict, optional):
                Either a :class:`NeighborList <freud.locality.NeighborList>` of
                neighbor pairs to use in the calculation, or a dictionary of
                `query arguments
                <https://freud.readthedocs.io/en/stable/topics/querying.html>`_
                (Default value: None).
        """
        cdef:
            freud.locality.NeighborQuery nq
            freud.locality.NeighborList nlist
            freud.locality._QueryArgs qargs
            const float[:, ::1] l_query_points
            unsigned int num_query_points

        nq, nlist, qargs, l_query_points, num_query_points = \
            self._preprocess_arguments(system, neighbors=neighbors)
        self.thisptr.compute(nlist.get_ptr(),
                             nq.get_ptr(),
                             dereference(qargs.thisptr))

    @property
    def l(self):  # noqa: E743
        """unsigned int: Spherical harmonic quantum number l."""
        return self.thisptr.getL()

    @property
    def q_threshold(self):
        """float: Value of dot product threshold."""
        return self.thisptr.getQThreshold()

    @property
    def solid_threshold(self):
        """float: Value of number-of-bonds threshold."""
        return self.thisptr.getSolidThreshold()

    @property
    def normalize_q(self):
        """bool: Whether the dot product is normalized."""
        return self.thisptr.getNormalizeQ()

    @_Compute._computed_property
    def cluster_idx(self):
        """:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`:
        Solid-like cluster indices for each particle."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getClusterIdx(),
            freud.util.arr_type_t.UNSIGNED_INT)

    @_Compute._computed_property
    def ql_ij(self):
        """:math:`\\left(N_{bonds}\\right)` :class:`numpy.ndarray`: Bond dot
        products :math:`q_l(i, j)`. Indexed by the elements of
        :code:`self.nlist`."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getQlij(),
            freud.util.arr_type_t.FLOAT)

    @_Compute._computed_property
    def cluster_sizes(self):
        """:math:`(N_{clusters}, )` :class:`np.ndarray`: The sizes of all
        clusters."""
        return np.asarray(self.thisptr.getClusterSizes())

    @_Compute._computed_property
    def largest_cluster_size(self):
        """unsigned int: The largest cluster size."""
        return self.thisptr.getLargestClusterSize()

    @_Compute._computed_property
    def nlist(self):
        """:class:`freud.locality.NeighborList`: Neighbor list of solid-like
        bonds."""
        return freud.locality._nlist_from_cnlist(self.thisptr.getNList())

    @_Compute._computed_property
    def num_connections(self):
        """:math:`\\left(N_{particles}\\right)` :class:`numpy.ndarray`: The
        number of solid-like bonds for each particle."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getNumberOfConnections(),
            freud.util.arr_type_t.UNSIGNED_INT)

    def __repr__(self):
        return ("freud.order.{cls}(l={sph_l}, q_threshold={q_threshold}, "
                "solid_threshold={solid_threshold}, "
                "normalize_q={normalize_q})").format(
                    cls=type(self).__name__,
                    sph_l=self.l,
                    q_threshold=self.q_threshold,
                    solid_threshold=self.solid_threshold,
                    normalize_q=self.normalize_q)

    def plot(self, ax=None):
        """Plot solid-like cluster distribution.

        Args:
            ax (:class:`matplotlib.axes.Axes`, optional): Axis to plot on. If
                :code:`None`, make a new figure and axis
                (Default value = :code:`None`).

        Returns:
            (:class:`matplotlib.axes.Axes`): Axis with the plot.
        """
        import freud.plot
        try:
            values, counts = np.unique(self.cluster_idx, return_counts=True)
        except ValueError:
            return None
        else:
            return freud.plot.clusters_plot(
                values, counts, num_clusters_to_plot=10, ax=ax)

    def _repr_png_(self):
        try:
            import freud.plot
            return freud.plot._ax_to_bytes(self.plot())
        except (AttributeError, ImportError):
            return None


cdef class RotationalAutocorrelation(_Compute):
    """Calculates a measure of total rotational autocorrelation.

    For any calculation of rotational correlations of extended (i.e. non-point)
    particles, encoding the symmetries of these particles is crucial to
    appropriately determining correlations. For systems of anisotropic
    particles in three dimensions, representing such equivalence can be quite
    mathematically challenging. This calculation is based on the hyperspherical
    harmonics as laid out in :cite:`Karas2019`. Generalizations of spherical
    harmonics to four dimensions, hyperspherical harmonics provide a natural
    basis for periodic functions in 4 dimensions just as harmonic functions
    (sines and cosines) or spherical harmonics do in lower dimensions. The idea
    behind this calculation is to embed orientation quaternions into a
    4-dimensional space and then use hyperspherical harmonics to find
    correlations in a symmetry-aware fashion.

    The choice of the hyperspherical harmonic parameter :math:`l` determines
    the symmetry of the functions. The output is not a correlation function,
    but rather a scalar value that measures total system orientational
    correlation with an initial state. As such, the output can be treated as an
    order parameter measuring degrees of rotational (de)correlation. For
    analysis of a trajectory, the compute call needs to be
    done at each trajectory frame.

    Args:
        l (int):
            Order of the hyperspherical harmonic. Must be a positive, even
            integer.
    """
    cdef freud._order.RotationalAutocorrelation * thisptr

    def __cinit__(self, l):
        if l % 2 or l < 0:
            raise ValueError(
                "The quantum number must be a positive, even integer.")
        self.thisptr = new freud._order.RotationalAutocorrelation(l)

    def __dealloc__(self):
        del self.thisptr

    def compute(self, ref_orientations, orientations):
        """Calculates the rotational autocorrelation function for a single frame.

        Args:
            ref_orientations ((:math:`N_{orientations}`, 4) :class:`numpy.ndarray`):
                Orientations for the initial frame.
            orientations ((:math:`N_{orientations}`, 4) :class:`numpy.ndarray`):
                Orientations for the frame of interest.
        """  # noqa
        ref_orientations = freud.util._convert_array(
            ref_orientations, shape=(None, 4))
        orientations = freud.util._convert_array(
            orientations, shape=ref_orientations.shape)

        cdef const float[:, ::1] l_ref_orientations = ref_orientations
        cdef const float[:, ::1] l_orientations = orientations
        cdef unsigned int nP = orientations.shape[0]

        self.thisptr.compute(
            <quat[float]*> &l_ref_orientations[0, 0],
            <quat[float]*> &l_orientations[0, 0],
            nP)
        return self

    @_Compute._computed_property
    def order(self):
        """float: Autocorrelation of the system."""
        return self.thisptr.getRotationalAutocorrelation()

    @_Compute._computed_property
    def particle_order(self):
        """(:math:`N_{orientations}`) :class:`numpy.ndarray`: Rotational
        autocorrelation values calculated for each orientation."""
        return freud.util.make_managed_numpy_array(
            &self.thisptr.getRAArray(),
            freud.util.arr_type_t.COMPLEX_FLOAT)

    @property
    def l(self):  # noqa: E743
        """int: The azimuthal quantum number, which defines the order of the
        hyperspherical harmonic."""
        return self.thisptr.getL()

    def __repr__(self):
        return "freud.order.{cls}(l={sph_l})".format(cls=type(self).__name__,
                                                     sph_l=self.l)
