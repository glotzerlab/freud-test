import numpy as np
import numpy.testing as npt
import freud
import matplotlib
import unittest
import warnings
import util
import rowan
from test_managedarray import TestManagedArray
matplotlib.use('agg')


TWO_PI = 2*np.pi


def build_radii(bin_centers):
    """Given bin centers in Cartesian coordinates, calculate distances from the
    origin."""
    radii = np.zeros([len(b) for b in bin_centers])
    for i, centers in enumerate(bin_centers):
        sl = tuple(slice(None, None, None) if i == j else None
                   for j in range(len(bin_centers)))
        radii += (centers**2)[sl]
    return np.sqrt(radii)


def pmft_to_rdf(pmft, radii):
    """Convert a PMFT to an RDF."""
    r_max = np.sqrt(sum(x[1]**2 for x in pmft.bounds))
    bin_edges = np.linspace(0, r_max, 100)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    rdf = np.zeros_like(bin_centers)

    with warnings.catch_warnings():
        # Ignore div by 0 and empty slice warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, (left, right) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            subset = np.where(np.logical_and(left < radii, radii < right))
            rdf[i] = np.mean(pmft._pcf[subset])

    return rdf


class TestPMFT:
    @classmethod
    def get_cubic_box(cls, L, ndim=None):
        if ndim is None:
            ndim = cls.ndim

        if ndim == 2:
            return freud.box.Box.square(L)
        else:
            return freud.box.Box.cube(L)

    @classmethod
    def make_pmft(cls):
        """Create a PMFT object."""
        return cls.pmft_cls(*cls.limits, bins=cls.bins)

    def test_box(self):
        (box, points), orientations = self.make_two_particle_system()
        pmft = self.make_pmft()
        pmft.compute((box, points), orientations)
        npt.assert_equal(pmft.box, self.get_cubic_box(self.L))

        # Ensure expected errors are raised
        box = self.get_cubic_box(self.L, ndim=2 if self.ndim == 3 else 3)
        with self.assertRaises(ValueError):
            pmft.compute((box, points), orientations)

    def get_bin_centers(self):
        edges = self.get_bin_edges()
        return [(edge[1:] + edge[:-1])/2 for edge in edges]

    def test_bins(self):
        bin_edges = self.get_bin_edges()
        bin_centers = self.get_bin_centers()

        pmft = self.make_pmft()

        for i in range(len(pmft.bin_centers)):
            npt.assert_allclose(pmft.bin_edges[i],
                                bin_edges[i], atol=1e-3)
            npt.assert_allclose(pmft.bin_centers[i],
                                bin_centers[i], atol=1e-3)
            npt.assert_equal(self.bins[i], pmft.nbins[i])

    def test_attribute_access(self):
        pmft = self.make_pmft()
        orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)

        with self.assertRaises(AttributeError):
            pmft.bin_counts
        with self.assertRaises(AttributeError):
            pmft.box
        with self.assertRaises(AttributeError):
            pmft.pmft

        box = self.get_cubic_box(self.L)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
                          dtype=np.float32)
        pmft.compute((box, points), orientations, reset=False)
        pmft.bin_counts
        pmft.pmft
        pmft.box
        npt.assert_equal(pmft.bin_counts.shape, self.bins)
        npt.assert_equal(pmft.pmft.shape, self.bins)

        # Test computing wihth reset
        pmft.compute((box, points), orientations)
        pmft.bin_counts
        pmft.pmft
        pmft.box

    def test_two_particles(self):
        (box, points), orientations = self.make_two_particle_system()

        correct_bin_counts = np.zeros(self.bins, dtype=np.int32)
        correct_bin_counts[self.get_bin(
            points[0], points[1], orientations[0], orientations[1])] = 1
        correct_bin_counts[self.get_bin(
            points[1], points[0], orientations[1], orientations[0])] = 1
        absoluteTolerance = 0.1

        # No angular terms should have entries in the limits array, so this
        # should work in all cases.
        r_max = np.linalg.norm(self.limits)
        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, 'ball', r_max, 0, True)
        for nq, neighbors in test_set:
            pmft = self.make_pmft()
            pmft.compute(nq, orientations, neighbors=neighbors, reset=False)
            npt.assert_allclose(pmft.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

            # Test with resetting.
            pmft.compute(nq, orientations, neighbors=neighbors)
            npt.assert_allclose(pmft.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

            # Test with angles.
            pmft.compute(nq, rowan.geometry.angle(orientations),
                         neighbors=neighbors)
            npt.assert_allclose(pmft.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

    def test_repr(self):
        pmft = self.make_pmft()
        self.assertEqual(str(pmft), str(eval(repr(pmft))))

    def test_pcf(self):
        """Verify that integrating a PMFT to generate an RDF for an ideal gas
        produces approximately unity everywhere."""

        def get_pmft_bydist(r_max, nbins):
            """Get a PMFT with a specified radial cutoff."""
            limit = np.sqrt(r_max**2/self.ndim)
            return self.pmft_cls(*(limit, )*len(self.limits), bins=nbins)

        L = 10
        N = 2500
        nbins = 100
        r_max = 3

        system = freud.data.make_random_system(L, N, self.ndim == 2, seed=3)
        orientations = rowan.random.rand(N) if self.ndim == 3 else \
            np.random.rand(N)*2*np.pi

        pmft = get_pmft_bydist(r_max, nbins)
        pmft.compute(system, orientations)

        def get_bin_centers(pmft):
            if len(self.limits) > 1:
                return pmft.bin_centers[:len(self.limits)]
            else:
                return [pmft.bin_centers[0]]

        radii = build_radii(get_bin_centers(pmft))
        rdf = pmft_to_rdf(pmft, radii)

        assert np.isclose(np.nanmean(rdf), 1, rtol=1e-2, atol=1e-2)


class TestPMFT2D(TestPMFT):
    def test_2d_box_3d_points(self):
        """Test that points with z != 0 fail if the box is 2D."""
        (box, points), orientations = self.make_two_particle_system()
        points[:, 2] = 1
        pmft = self.make_pmft()
        with self.assertRaises(ValueError):
            pmft.compute((box, points), orientations,
                         neighbors={'mode': 'nearest', 'num_neighbors': 1})


class TestPMFTR12(TestPMFT2D, unittest.TestCase):
    limits = (5.23, )
    bins = (10, 20, 30)
    ndim = 2
    L = 16
    pmft_cls = freud.pmft.PMFTR12

    def get_bin_edges(self):
        # make sure the radius for each bin is generated correctly
        listR = np.linspace(0, self.limits[0], self.bins[0] + 1,
                            dtype=np.float32)
        listT1 = np.linspace(0, 2*np.pi, self.bins[1] + 1, dtype=np.float32)
        listT2 = np.linspace(0, 2*np.pi, self.bins[2] + 1, dtype=np.float32)
        return (listR, listT1, listT2)

    def make_two_particle_system(self):
        box = self.get_cubic_box(self.L)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
                          dtype=np.float32)
        orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)

        return (box, points), orientations

    def get_bin(self, query_point, point, query_point_orientation,
                point_orientation):
        query_point_angle = rowan.geometry.angle(query_point_orientation)
        point_angle = rowan.geometry.angle(point_orientation)

        r_ij = point - query_point
        r_bin = np.floor(np.linalg.norm(r_ij) * self.bins[0] / self.limits[0]
                         ).astype(int)

        delta_t1 = np.arctan2(r_ij[1], r_ij[0])
        t1_bin = np.floor(((point_angle - delta_t1) % TWO_PI) * self.bins[1] /
                          TWO_PI).astype(int)

        delta_t2 = np.arctan2(-r_ij[1], -r_ij[0])
        t2_bin = np.floor(((query_point_angle - delta_t2) % TWO_PI) *
                          self.bins[2] / TWO_PI).astype(int)
        return (r_bin, t1_bin, t2_bin)

    def test_points_ne_query_points(self):
        r_max = 2.3
        nbins = 10

        lattice_size = 10
        box = freud.box.Box.square(lattice_size*5)

        points, query_points = util.make_alternating_lattice(
            lattice_size, 0.01, 2)
        orientations = np.array([0]*len(points))
        query_orientations = np.array([0]*len(query_points))

        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, "ball", r_max, 0, False)
        for nq, neighbors in test_set:
            pmft = freud.pmft.PMFTR12(r_max, nbins)
            pmft.compute(nq,
                         orientations, query_points,
                         query_orientations, neighbors=neighbors)

            self.assertEqual(np.count_nonzero(np.isinf(pmft.pmft) == 0), 12)
            self.assertEqual(len(np.unique(pmft.pmft)), 3)


class TestPMFTXYT(TestPMFT2D, unittest.TestCase):
    limits = (3.6, 4.2)
    bins = (20, 30, 40)
    ndim = 2
    L = 16
    pmft_cls = freud.pmft.PMFTXYT

    def get_bin_edges(self):
        listX = np.linspace(-self.limits[0], self.limits[0], self.bins[0] + 1,
                            dtype=np.float32)
        listY = np.linspace(-self.limits[1], self.limits[1], self.bins[1] + 1,
                            dtype=np.float32)
        listT = np.linspace(0, 2*np.pi, self.bins[2] + 1, dtype=np.float32)
        return (listX, listY, listT)

    def make_two_particle_system(self):
        box = self.get_cubic_box(self.L)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
                          dtype=np.float32)
        orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)

        return (box, points), orientations

    def get_bin(self, query_point, point, query_point_orientation,
                point_orientation):
        r_ij = point - query_point
        rot_r_ij = rowan.rotate(rowan.conjugate(query_point_orientation), r_ij)

        limits = np.asarray(self.limits)
        xy_bins = tuple(np.floor((
            rot_r_ij[:2] + limits) * np.asarray(self.bins[:2]) /
            (2*limits)).astype(np.int32))

        point_angle = rowan.geometry.angle(point_orientation)
        angle_bin = np.floor(
            ((point_angle - np.arctan2(-r_ij[1], -r_ij[0])) %
                TWO_PI) * self.bins[2] / TWO_PI).astype(np.int32)
        return xy_bins + (angle_bin, )

    def test_points_ne_query_points(self):
        x_max = 2.5
        y_max = 2.5
        n_x = 10
        n_y = 10
        n_t = 4

        lattice_size = 10
        box = freud.box.Box.square(lattice_size*5)

        points, query_points = util.make_alternating_lattice(
            lattice_size, 0.01, 2)
        orientations = np.array([0]*len(points))
        query_orientations = np.array([0]*len(query_points))

        r_max = np.sqrt(x_max**2 + y_max**2)
        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, 'ball', r_max, 0, False)

        for nq, neighbors in test_set:
            pmft = freud.pmft.PMFTXYT(x_max, y_max, (n_x, n_y, n_t))
            pmft.compute(nq,
                         orientations, query_points,
                         query_orientations, neighbors=neighbors)

            # when rotated slightly, for each ref point, each quadrant
            # (corresponding to two consecutive bins) should contain 3 points.
            for i in range(n_t):
                self.assertEqual(
                    np.count_nonzero(np.isinf(pmft.pmft[..., i]) == 0), 3)

            self.assertEqual(len(np.unique(pmft.pmft)), 2)

    def test_nontrivial_orientations(self):
        """Ensure that orientations are applied to the right particles."""
        box = self.get_cubic_box(6)
        points = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        query_points = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)

        for angles in ([0], [np.pi/4]):
            for query_angles in ([0.01], [np.pi/4 + 0.01]):
                max_width = 2
                nbins = 4
                self.limits = (max_width, )*2
                self.bins = (nbins, nbins, nbins)

                pmft = freud.pmft.PMFTXYT(max_width, max_width, nbins)
                pmft.compute((box, points), angles, query_points, query_angles)

                query_orientation = rowan.from_axis_angle(
                    [0, 0, 1], query_angles[0])
                orientation = rowan.from_axis_angle([0, 0, 1], angles[0])

                self.assertEqual(
                    tuple(np.asarray(np.where(pmft.bin_counts)).flatten()),
                    self.get_bin(query_points[0], points[0],
                                 query_orientation, orientation)
                )
                self.assertEqual(np.sum(pmft.bin_counts), 1)


class TestPMFTXY(TestPMFT2D, unittest.TestCase):
    limits = (3.6, 4.2)
    bins = (100, 110)
    ndim = 2
    L = 16
    pmft_cls = freud.pmft.PMFTXY

    def get_bin_edges(self):
        listX = np.linspace(-self.limits[0], self.limits[0], self.bins[0] + 1,
                            dtype=np.float32)
        listY = np.linspace(-self.limits[1], self.limits[1], self.bins[1] + 1,
                            dtype=np.float32)
        return (listX, listY)

    def make_two_particle_system(self):
        box = self.get_cubic_box(self.L)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
                          dtype=np.float32)
        orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)

        return (box, points), orientations

    def get_bin(self, query_point, point, query_point_orientation,
                point_orientation):
        r_ij = point - query_point
        rot_r_ij = rowan.rotate(rowan.conjugate(query_point_orientation), r_ij)

        limits = np.asarray(self.limits)
        return tuple(np.floor((
            rot_r_ij[:2] + limits) * np.asarray(self.bins) /
            (2*limits)).astype(np.int32))

    def test_repr_png(self):
        L = 16.0
        box = freud.box.Box.square(L)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                          dtype=np.float32)
        angles = np.array([0.0, 0.0], dtype=np.float32)
        max_x = 3.6
        max_y = 4.2
        nbins_x = 100
        nbins_y = 110
        pmft = freud.pmft.PMFTXY(max_x, max_y, (nbins_x, nbins_y))

        with self.assertRaises(AttributeError):
            pmft.plot()
        self.assertEqual(pmft._repr_png_(), None)

        pmft.compute((box, points), angles, points, reset=False)
        pmft._repr_png_()

    def test_points_ne_query_points(self):
        x_max = 2.5
        y_max = 2.5
        nbins = 20

        lattice_size = 10
        box = freud.box.Box.square(lattice_size*5)

        points, query_points = util.make_alternating_lattice(
            lattice_size, 0.01, 2)

        query_orientations = np.array([0]*len(query_points))

        r_max = np.sqrt(x_max**2 + y_max**2)
        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, 'ball', r_max, 0, False)

        for nq, neighbors in test_set:
            pmft = freud.pmft.PMFTXY(x_max, y_max, nbins)
            pmft.compute(
                nq, query_orientations, query_points, neighbors)

            self.assertEqual(np.count_nonzero(np.isinf(pmft.pmft) == 0), 12)
            self.assertEqual(len(np.unique(pmft.pmft)), 2)

    def test_query_args_nn(self):
        """Test that using nn based query args works."""
        L = 8
        box = freud.box.Box.square(L)
        points = np.array([[0, 0, 0]],
                          dtype=np.float32)
        query_points = np.array([[1.1, 0.0, 0.0],
                                [-1.2, 0.0, 0.0],
                                [0.0, 1.3, 0.0],
                                [0.0, -1.4, 0.0]],
                                dtype=np.float32)
        angles = np.array([0.0]*points.shape[0], dtype=np.float32)
        query_angles = np.array([0.0]*query_points.shape[0], dtype=np.float32)

        max_width = 3
        nbins = 3
        pmft = freud.pmft.PMFTXY(max_width, max_width, nbins)
        pmft.compute((box, points), query_angles, query_points,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        # Now every point in query_points will find the origin as a neighbor.
        npt.assert_array_equal(
            pmft.bin_counts,
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]])
        # Now there will be only one neighbor for the single point.
        pmft.compute((box, query_points), angles, points,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(
            pmft.bin_counts,
            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]])

    def test_orientation_with_query_points(self):
        """The orientations should be associated with the query points if they
        are provided."""
        L = 8
        box = freud.box.Box.square(L)
        # Don't place the points at exactly distances of 0/1 apart to avoid any
        # ambiguity when the distances fall on the bin boundaries.
        points = np.array([[0.1, 0.1, 0]],
                          dtype=np.float32)
        points2 = np.array([[1, 0, 0]],
                           dtype=np.float32)
        angles = np.array([np.deg2rad(0)]*points2.shape[0], dtype=np.float32)

        max_width = 3
        cells_per_unit_length = 4
        nbins = max_width * cells_per_unit_length
        pmft = freud.pmft.PMFTXY(max_width, max_width, nbins)

        # In this case, the only nonzero bin should be in the bin corresponding
        # to dx=-0.9, dy=0.1, which is (4, 6).
        pmft.compute((box, points), angles, points2,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(
            np.asarray(np.where(pmft.bin_counts)).squeeze(),
            (4, 6))

        # Now the sets of points are swapped, so dx=0.9, dy=-0.1, which is
        # (7, 5).
        pmft.compute((box, points2), angles, points,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(
            np.asarray(np.where(pmft.bin_counts)).squeeze(),
            (7, 5))

        # Apply a rotation to whichever point is provided as a query_point by
        # 45 degrees (easiest to picture if you think of each point as a
        # square).
        angles = np.array([np.deg2rad(45)]*points2.shape[0], dtype=np.float32)

        # Determine the relative position of the point when points2 is rotated
        # by 45 degrees. Since we're undoing the orientation of the orientation
        # of the particle, we have to conjugate the quaternion.
        quats = rowan.from_axis_angle([0, 0, 1], angles)
        bond_vector = rowan.rotate(rowan.conjugate(quats), points - points2)
        bins = ((bond_vector+max_width)*cells_per_unit_length/2).astype(int)

        pmft.compute((box, points), angles, points2,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(
            np.asarray(np.where(pmft.bin_counts)).squeeze(),
            bins.squeeze()[:2])

        # If we swap the order of the points, the angle should no longer
        # matter.
        bond_vector = rowan.rotate(rowan.conjugate(quats), points2 - points)
        bins = ((bond_vector+max_width)*cells_per_unit_length/2).astype(int)

        pmft.compute((box, points2), angles, points,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(
            np.asarray(np.where(pmft.bin_counts)).squeeze(),
            bins.squeeze()[:2])

    def test_orientation_with_fewer_query_points(self):
        """The orientations should be associated with the query points if they
        are provided. Ensure that this works when the number of points and
        query points differ."""
        L = 8
        box = freud.box.Box.square(L)
        # Don't place the points at exactly distances of 0/1 apart to avoid any
        # ambiguity when the distances fall on the bin boundaries.
        points = np.array([[0.1, 0.1, 0], [0.89, 0.89, 0]],
                          dtype=np.float32)
        points2 = np.array([[1, 0, 0]],
                           dtype=np.float32)
        angles = np.array([np.deg2rad(0)]*points.shape[0], dtype=np.float32)
        angles2 = np.array([np.deg2rad(0)]*points2.shape[0], dtype=np.float32)

        def points_to_set(bin_counts):
            """Extract set of unique occupied bins from pmft bin counts."""
            return set(zip(*np.asarray(np.where(bin_counts)).tolist()))

        max_width = 3
        cells_per_unit_length = 4
        nbins = max_width * cells_per_unit_length
        pmft = freud.pmft.PMFTXY(max_width, max_width, nbins)

        # There should be two nonzero bins:
        #     dx=-0.9, dy=0.1: bin (4, 6).
        #     dx=-0.11, dy=0.89: bin (5, 7).
        pmft.compute((box, points), angles2, points2,
                     neighbors={'mode': 'nearest', 'num_neighbors': 2})
        npt.assert_array_equal(
            points_to_set(pmft.bin_counts),
            set([(4, 6), (5, 7)]))

        # Now the sets of points are swapped, so:
        #     dx=0.9, dy=-0.1: bin (7, 5).
        #     dx=0.11, dy=-0.89: bin (6, 4).
        pmft.compute((box, points2), angles, points,
                     neighbors={'mode': 'nearest', 'num_neighbors': 2})
        npt.assert_array_equal(
            points_to_set(pmft.bin_counts),
            set([(7, 5), (6, 4)]))

        # Apply a rotation to whichever point is provided as a query_point by
        # 45 degrees (easiest to picture if you think of each point as a
        # square).
        angles2 = np.array([np.deg2rad(45)]*points2.shape[0], dtype=np.float32)

        # Determine the relative position of the point when points2 is rotated
        # by 45 degrees. Since we're undoing the orientation of the orientation
        # of the particle, we have to conjugate the quaternion.
        quats = rowan.from_axis_angle([0, 0, 1], angles2)
        bond_vector = rowan.rotate(rowan.conjugate(quats), points - points2)
        bins = ((bond_vector+max_width)*cells_per_unit_length/2).astype(int)
        bins = set([tuple(x) for x in bins[:, :2]])

        pmft.compute((box, points), angles2, points2,
                     neighbors={'mode': 'nearest', 'num_neighbors': 2})
        npt.assert_array_equal(
            points_to_set(pmft.bin_counts),
            bins)


class TestPMFTXYZ(TestPMFT, unittest.TestCase):
    limits = (5.23, 6.23, 7.23)
    bins = (100, 110, 120)
    ndim = 3
    L = 25
    pmft_cls = freud.pmft.PMFTXYZ

    def get_bin_edges(self):
        listX = np.linspace(-self.limits[0], self.limits[0], self.bins[0] + 1,
                            dtype=np.float32)
        listY = np.linspace(-self.limits[1], self.limits[1], self.bins[1] + 1,
                            dtype=np.float32)
        listZ = np.linspace(-self.limits[2], self.limits[2], self.bins[2] + 1,
                            dtype=np.float32)
        return (listX, listY, listZ)

    def get_bin(self, query_point, point, query_point_orientation,
                point_orientation):
        r_ij = point - query_point
        rot_r_ij = rowan.rotate(point_orientation, r_ij)

        limits = np.asarray(self.limits)
        return tuple(np.floor((
            rot_r_ij + limits) * np.asarray(self.bins) /
            (2*limits)).astype(np.int32))

    def make_two_particle_system(self):
        box = self.get_cubic_box(self.L)
        points = np.array([[-1.0, 0.0, 0.0], [1.0, 0.1, 0.0]],
                          dtype=np.float32)
        orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)

        return (box, points), orientations

    def test_two_particles(self):
        """Override base class function to test equivalent orientations."""
        (box, points), orientations = self.make_two_particle_system()

        correct_bin_counts = np.zeros(self.bins, dtype=np.int32)
        correct_bin_counts[self.get_bin(
            points[0], points[1], orientations[0], orientations[1])] = 1
        correct_bin_counts[self.get_bin(
            points[1], points[0], orientations[1], orientations[0])] = 1
        absoluteTolerance = 0.1

        # No angular terms should have entries in the limits array, so this
        # should work in all cases.
        r_max = np.linalg.norm(self.limits)
        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, 'ball', r_max, 0, True)
        for nq, neighbors in test_set:
            pmft = self.make_pmft()
            pmft.compute(nq, orientations, neighbors=neighbors, reset=False)
            npt.assert_allclose(pmft.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)

            # Test with resetting.
            pmft.compute(nq, orientations, neighbors=neighbors)
            npt.assert_allclose(pmft.bin_counts, correct_bin_counts,
                                atol=absoluteTolerance)
            orig_pmft = pmft.pmft

            # Test with equivalent orientations.
            pmft.compute(nq, orientations, neighbors=neighbors,
                         equiv_orientations=[[1, 0, 0, 0]]*2)
            npt.assert_allclose(pmft.bin_counts, 2*correct_bin_counts,
                                atol=absoluteTolerance)
            npt.assert_allclose(pmft.pmft, orig_pmft,
                                atol=absoluteTolerance)

    def test_shift_two_particles_dead_pixel(self):
        points = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float32)
        query_orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]],
                                      dtype=np.float32)
        noshift = freud.pmft.PMFTXYZ(0.5, 0.5, 0.5, 3, shiftvec=[0, 0, 0])
        shift = freud.pmft.PMFTXYZ(0.5, 0.5, 0.5, 3, shiftvec=[1, 1, 1])

        for pm in [noshift, shift]:
            pm.compute((freud.box.Box.cube(3), points), query_orientations)

        # Ignore warnings about NaNs
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Non-shifted pmft should have no non-inf valued voxels,
        # since the other point is outside the x/y/z max
        infcheck_noshift = np.isfinite(noshift.pmft).sum()
        # Shifted pmft should have one non-inf valued voxel
        infcheck_shift = np.isfinite(shift.pmft).sum()

        npt.assert_equal(infcheck_noshift, 0)
        npt.assert_equal(infcheck_shift, 1)

    def test_query_args_nn(self):
        """Test that using nn based query args works."""
        L = 8
        box = freud.box.Box.cube(L)
        points = np.array([[0, 0, 0]],
                          dtype=np.float32)
        query_points = np.array([[1.1, 0.0, 0.0],
                                [-1.2, 0.0, 0.0],
                                [0.0, 1.3, 0.0],
                                [0.0, -1.4, 0.0],
                                [0.0, 0.0, 1.5],
                                [0.0, 0.0, -1.6]],
                                dtype=np.float32)
        angles = np.array([[1.0, 0.0, 0.0, 0.0]]*points.shape[0],
                          dtype=np.float32)
        query_angles = np.array([[1.0, 0.0, 0.0, 0.0]]*query_points.shape[0],
                                dtype=np.float32)

        max_width = 3
        nbins = 3
        pmft = freud.pmft.PMFTXYZ(max_width, max_width, max_width, nbins)
        pmft.compute((box, points), query_angles, query_points,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})

        # Now every point in query_points will find the origin as a neighbor.
        npt.assert_array_equal(
            pmft.bin_counts,
            [[[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]],
             [[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]],
             [[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]]])

        pmft.compute((box, query_points), angles, points,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        # The only nonzero bin is the right-center bin (zero distance in y, z)
        self.assertEqual(pmft.bin_counts[2, 1, 1], 1)
        self.assertEqual(np.sum(pmft.bin_counts), 1)
        self.assertTrue(np.all(pmft.bin_counts >= 0))

    def test_orientation_with_query_points(self):
        """The orientations should be associated with the query points if they
        are provided."""
        L = 8
        box = freud.box.Box.cube(L)
        # Don't place the points at exactly distances of 0/1 apart to avoid any
        # ambiguity when the distances fall on the bin boundaries.
        points = np.array([[0.1, 0.1, 0]],
                          dtype=np.float32)
        points2 = np.array([[1, 0, 0]],
                           dtype=np.float32)
        angles = np.array([np.deg2rad(0)]*points2.shape[0], dtype=np.float32)
        quats = rowan.from_axis_angle([0, 0, 1], angles)

        max_width = 3
        cells_per_unit_length = 4
        nbins = max_width * cells_per_unit_length
        pmft = freud.pmft.PMFTXYZ(max_width, max_width, max_width, nbins)

        # In this case, the only nonzero bin should be in the bin corresponding
        # to dx=-0.9, dy=0.1, which is (4, 6).
        pmft.compute((box, points), quats, points2,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(
            np.asarray(np.where(pmft.bin_counts)).squeeze(),
            (4, 6, 6))

        # Now the sets of points are swapped, so dx=0.9, dy=-0.1, which is
        # (7, 5).
        pmft.compute((box, points2), quats, points,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(
            np.asarray(np.where(pmft.bin_counts)).squeeze(),
            (7, 5, 6))

        # Apply a rotation to whichever point is provided as a query_point by
        # 45 degrees (easiest to picture if you think of each point as a
        # square).
        angles = np.array([np.deg2rad(45)]*points2.shape[0], dtype=np.float32)
        quats = rowan.from_axis_angle([0, 0, 1], angles)

        # Determine the relative position of the point when points2 is rotated
        # by 45 degrees. Since we're undoing the orientation of the orientation
        # of the particle, we have to conjugate the quaternion.
        bond_vector = rowan.rotate(rowan.conjugate(quats), points - points2)
        bins = ((bond_vector+max_width)*cells_per_unit_length/2).astype(int)

        pmft.compute((box, points), quats, points2,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(
            np.asarray(np.where(pmft.bin_counts)).squeeze(),
            bins.squeeze())

        # If we swap the order of the points, the angle should no longer
        # matter.
        bond_vector = rowan.rotate(rowan.conjugate(quats), points2 - points)
        bins = ((bond_vector+max_width)*cells_per_unit_length/2).astype(int)

        pmft.compute((box, points2), quats, points,
                     neighbors={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(
            np.asarray(np.where(pmft.bin_counts)).squeeze(),
            bins.squeeze())

    def test_orientation_with_fewer_query_points(self):
        """The orientations should be associated with the query points if they
        are provided. Ensure that this works when the number of points and
        query points differ."""
        L = 8
        box = freud.box.Box.cube(L)
        # Don't place the points at exactly distances of 0/1 apart to avoid any
        # ambiguity when the distances fall on the bin boundaries.
        points = np.array([[0.1, 0.1, 0], [0.89, 0.89, 0]],
                          dtype=np.float32)
        points2 = np.array([[1, 0, 0]],
                           dtype=np.float32)
        angles = np.array([np.deg2rad(0)]*points.shape[0], dtype=np.float32)
        quats = rowan.from_axis_angle([0, 0, 1], angles)
        angles2 = np.array([np.deg2rad(0)]*points2.shape[0], dtype=np.float32)
        quats2 = rowan.from_axis_angle([0, 0, 1], angles2)

        def points_to_set(bin_counts):
            """Extract set of unique occupied bins from pmft bin counts."""
            return set(zip(*np.asarray(np.where(bin_counts)).tolist()))

        max_width = 3
        cells_per_unit_length = 4
        nbins = max_width * cells_per_unit_length
        pmft = freud.pmft.PMFTXYZ(max_width, max_width, max_width, nbins)

        # There should be two nonzero bins:
        #     dx=-0.9, dy=0.1: bin (4, 6).
        #     dx=-0.11, dy=0.89: bin (5, 7).
        pmft.compute((box, points), quats2, points2,
                     neighbors={'mode': 'nearest', 'num_neighbors': 2})
        npt.assert_array_equal(
            points_to_set(pmft.bin_counts),
            set([(4, 6, 6), (5, 7, 6)]))

        # Now the sets of points are swapped, so:
        #     dx=0.9, dy=-0.1: bin (7, 5).
        #     dx=0.11, dy=-0.89: bin (6, 4).
        pmft.compute((box, points2), quats, points,
                     neighbors={'mode': 'nearest', 'num_neighbors': 2})
        npt.assert_array_equal(
            points_to_set(pmft.bin_counts),
            set([(7, 5, 6), (6, 4, 6)]))

        # Apply a rotation to whichever point is provided as a query_point by
        # 45 degrees (easiest to picture if you think of each point as a
        # square).
        angles2 = np.array([np.deg2rad(45)]*points2.shape[0], dtype=np.float32)
        quats2 = rowan.from_axis_angle([0, 0, 1], angles2)

        # Determine the relative position of the point when points2 is rotated
        # by 45 degrees. Since we're undoing the orientation of the orientation
        # of the particle, we have to conjugate the quaternion.
        bond_vector = rowan.rotate(rowan.conjugate(quats2), points - points2)
        bins = ((bond_vector+max_width)*cells_per_unit_length/2).astype(int)
        bins = set([tuple(x) for x in bins])

        pmft.compute((box, points), quats2, points2,
                     neighbors={'mode': 'nearest', 'num_neighbors': 2})
        npt.assert_array_equal(
            points_to_set(pmft.bin_counts),
            bins)


class TestPMFTR12ManagedArray(TestManagedArray, unittest.TestCase):
    def build_object(self):
        self.obj = freud.pmft.PMFTR12(5, (50, 50, 50))

    @property
    def computed_properties(self):
        return ['bin_counts', 'pmft']

    def compute(self):
        box = freud.box.Box.square(10)
        num_points = 100
        points = np.random.rand(
            num_points, 3)*box.L - box.L/2
        angles = np.random.rand(num_points)*2*np.pi
        self.obj.compute((box, points), angles, neighbors={'r_max': 3})


class TestPMFTXYTManagedArray(TestManagedArray, unittest.TestCase):
    def build_object(self):
        self.obj = freud.pmft.PMFTXYT(5, 5, (50, 50, 50))

    @property
    def computed_properties(self):
        return ['bin_counts', 'pmft']

    def compute(self):
        box = freud.box.Box.square(10)
        num_points = 100
        points = np.random.rand(
            num_points, 3)*box.L - box.L/2
        angles = np.random.rand(num_points)*2*np.pi
        self.obj.compute((box, points), angles, neighbors={'r_max': 3})


class TestPMFTXYManagedArray(TestManagedArray, unittest.TestCase):
    def build_object(self):
        self.obj = freud.pmft.PMFTXY(5, 5, (50, 50))

    @property
    def computed_properties(self):
        return ['bin_counts', 'pmft']

    def compute(self):
        box = freud.box.Box.square(10)
        num_points = 100
        points = np.random.rand(
            num_points, 3)*box.L - box.L/2
        angles = np.random.rand(num_points)*2*np.pi
        self.obj.compute((box, points), angles, neighbors={'r_max': 3})


class TestPMFTXYZManagedArray(TestManagedArray, unittest.TestCase):
    def build_object(self):
        self.obj = freud.pmft.PMFTXYZ(5, 5, 5, (50, 50, 50))

    @property
    def computed_properties(self):
        return ['bin_counts', 'pmft']

    def compute(self):
        box = freud.box.Box.cube(10)
        num_points = 100
        points = np.random.rand(
            num_points, 3)*box.L - box.L/2
        orientations = rowan.random.rand(num_points)
        self.obj.compute((box, points), orientations, neighbors={'r_max': 3})


class TestCompare(unittest.TestCase):
    def test_XY_XYZ(self):
        """Check that 2D and 3D PMFTs give the same results."""
        x_max = 2.5
        y_max = 2.5
        z_max = 1
        nbins = 4
        num_points = 100
        L = 10

        box2d = freud.box.Box.square(L)
        box3d = freud.box.Box.cube(L)

        points = np.random.rand(num_points, 3)
        points[:, 2] = 0
        orientations = np.array([[1, 0, 0, 0]]*len(points))

        pmft2d = freud.pmft.PMFTXY(x_max, y_max, nbins)
        pmft2d.compute((box2d, points), orientations)

        pmft3d = freud.pmft.PMFTXYZ(x_max, y_max, z_max, nbins)
        pmft3d.compute((box3d, points), orientations)

        # Bin counts are equal, PMFTs are scaled by the box length in z.
        npt.assert_array_equal(pmft2d.bin_counts,
                               pmft3d.bin_counts[:, :, nbins//2])
        # The numerator of the scale factor comes from the extra z bins (which
        # we cannot avoid adding because of the query distance limitations on
        # NeighborQuery objects). The denominator comes from the 8pi^2 of
        # orientational phase space in PMFTXYZ divided by the 2pi in theta
        # space in PMFTXY.
        scale_factor = ((nbins/2)*L)
        npt.assert_allclose(np.exp(pmft2d.pmft),
                            np.exp(pmft3d.pmft[:, :, nbins//2])*scale_factor,
                            atol=1e-6)

    def test_XY_XYT(self):
        """Check that XY and XYT PMFTs give the same results."""
        x_max = 2.5
        y_max = 2.5
        nbins = 3
        nbinsxyt = (3, 3, 1)
        num_points = 100
        L = 10

        box = freud.box.Box.square(L)

        np.random.seed(0)
        points = np.random.rand(num_points, 3)
        points[:, 2] = 0
        orientations = np.array([0]*len(points))

        pmftxy = freud.pmft.PMFTXY(x_max, y_max, nbins)
        pmftxy.compute((box, points), orientations)

        pmftxyt = freud.pmft.PMFTXYT(x_max, y_max, nbinsxyt)
        pmftxyt.compute((box, points), orientations)

        npt.assert_array_equal(pmftxy.bin_counts,
                               pmftxyt.bin_counts.reshape(nbins, nbins))
        npt.assert_allclose(np.exp(pmftxy.pmft),
                            np.exp(pmftxyt.pmft).reshape(nbins, nbins),
                            atol=1e-6)


if __name__ == '__main__':
    unittest.main()
