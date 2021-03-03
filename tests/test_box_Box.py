import numpy as np
import numpy.testing as npt
import freud
from collections import namedtuple
import matplotlib
import unittest
import warnings
matplotlib.use('agg')


class TestBox(unittest.TestCase):

    def setUp(self):
        # We ignore warnings for test_2_dimensional
        warnings.simplefilter("ignore")

    def test_construct(self):
        """Test correct behavior for various constructor signatures"""
        with self.assertRaises(ValueError):
            freud.box.Box(0, 0)

        with self.assertRaises(ValueError):
            freud.box.Box(1, 2, is2D=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            freud.box.Box(1, 2, 3, is2D=True)
            self.assertTrue(len(w) == 1)

        box = freud.box.Box(1, 2)
        self.assertTrue(box.dimensions == 2)

    def test_get_length(self):
        box = freud.box.Box(2, 4, 5, 1, 0, 0)

        npt.assert_allclose(box.Lx, 2, rtol=1e-6)
        npt.assert_allclose(box.Ly, 4, rtol=1e-6)
        npt.assert_allclose(box.Lz, 5, rtol=1e-6)
        npt.assert_allclose(box.L, [2, 4, 5], rtol=1e-6)
        npt.assert_allclose(box.L_inv, [0.5, 0.25, 0.2], rtol=1e-6)

    def test_set_length(self):
        # Make sure we can change the lengths of the box after its creation
        box = freud.box.Box(1, 2, 3, 1, 0, 0)

        box.Lx = 4
        box.Ly = 5
        box.Lz = 6

        npt.assert_allclose(box.Lx, 4, rtol=1e-6)
        npt.assert_allclose(box.Ly, 5, rtol=1e-6)
        npt.assert_allclose(box.Lz, 6, rtol=1e-6)

        box.L = [7, 8, 9]
        npt.assert_allclose(box.L, [7, 8, 9], rtol=1e-6)

        with self.assertRaises(ValueError):
            box.L = [1, 2, 3, 4]

        with self.assertRaises(ValueError):
            box.L = [1, 2]

    def test_get_tilt_factor(self):
        box = freud.box.Box(2, 2, 2, 1, 2, 3)

        npt.assert_allclose(box.xy, 1, rtol=1e-6)
        npt.assert_allclose(box.xz, 2, rtol=1e-6)
        npt.assert_allclose(box.yz, 3, rtol=1e-6)

    def test_set_tilt_factor(self):
        box = freud.box.Box(2, 2, 2, 1, 2, 3)
        box.xy = 4
        box.xz = 5
        box.yz = 6

        npt.assert_allclose(box.xy, 4, rtol=1e-6)
        npt.assert_allclose(box.xz, 5, rtol=1e-6)
        npt.assert_allclose(box.yz, 6, rtol=1e-6)

    def test_box_volume(self):
        box3d = freud.box.Box(2, 2, 2, 1, 0, 0)
        box2d = freud.box.Box(2, 2, 0, 0, 0, 0, is2D=True)

        npt.assert_allclose(box3d.volume, 8, rtol=1e-6)
        npt.assert_allclose(box2d.volume, 4, rtol=1e-6)

    def test_wrap_single_particle(self):
        box = freud.box.Box(2, 2, 2, 1, 0, 0)

        points = [0, -1, -1]
        npt.assert_allclose(box.wrap(points)[0], -2, rtol=1e-6)

        points = np.array(points)
        npt.assert_allclose(box.wrap(points)[0], -2, rtol=1e-6)

        with self.assertRaises(ValueError):
            box.wrap([1, 2])

    def test_wrap_multiple_particles(self):
        box = freud.box.Box(2, 2, 2, 1, 0, 0)

        points = [[0, -1, -1], [0, 0.5, 0]]
        npt.assert_allclose(box.wrap(points)[0, 0], -2, rtol=1e-6)

        points = np.array(points)
        npt.assert_allclose(box.wrap(points)[0, 0], -2, rtol=1e-6)

    def test_wrap_multiple_images(self):
        box = freud.box.Box(2, 2, 2, 1, 0, 0)

        points = [[10, -5, -5], [0, 0.5, 0]]
        npt.assert_allclose(box.wrap(points)[0, 0], -2, rtol=1e-6)

        points = np.array(points)
        npt.assert_allclose(box.wrap(points)[0, 0], -2, rtol=1e-6)

    def test_unwrap(self):
        box = freud.box.Box(2, 2, 2, 1, 0, 0)

        points = [0, -1, -1]
        imgs = [1, 0, 0]
        npt.assert_allclose(
            box.unwrap(points, imgs), [2, -1, -1], rtol=1e-6)

        points = [[0, -1, -1], [0, 0.5, 0]]
        imgs = [[1, 0, 0], [1, 1, 0]]
        npt.assert_allclose(box.unwrap(points, imgs)[0, 0], 2, rtol=1e-6)

        points = np.array(points)
        imgs = np.array(imgs)
        npt.assert_allclose(box.unwrap(points, imgs)[0, 0], 2, rtol=1e-6)

        with self.assertRaises(ValueError):
            box.unwrap(points, imgs[..., np.newaxis])

        with self.assertRaises(ValueError):
            box.unwrap(points[:, :2], imgs)

        # Now test 2D
        box = freud.box.Box.square(1)

        points = [10, 0, 0]
        imgs = [10, 1, 2]
        npt.assert_allclose(
            box.unwrap(points, imgs), [20, 1, 0], rtol=1e-6)

        # Test broadcasting one image with multiple vectors
        box = freud.box.Box.cube(1)

        points = [[10, 0, 0], [11, 0, 0]]
        imgs = [10, 1, 2]
        npt.assert_allclose(
            box.unwrap(points, imgs), [[20, 1, 2], [21, 1, 2]], rtol=1e-6)

        # Test broadcasting one vector with multiple images
        box = freud.box.Box.cube(1)

        points = [10, 0, 0]
        imgs = [[10, 1, 2], [11, 1, 2]]
        npt.assert_allclose(
            box.unwrap(points, imgs), [[20, 1, 2], [21, 1, 2]], rtol=1e-6)

    def test_images_3d(self):
        box = freud.box.Box(2, 2, 2, 0, 0, 0)
        points = np.array([[50, 40, 30],
                           [-10, 0, 0]])
        images = np.array([box.get_images(vec) for vec in points])
        npt.assert_equal(images,
                         np.array([[25, 20, 15],
                                   [-5, 0, 0]]))
        images = box.get_images(points)
        npt.assert_equal(images,
                         np.array([[25, 20, 15],
                                   [-5, 0, 0]]))

    def test_images_2d(self):
        box = freud.box.Box(2, 2, 0, 0, 0, 0)
        points = np.array([[50, 40, 0],
                           [-10, 0, 0]])
        images = np.array([box.get_images(vec) for vec in points])
        npt.assert_equal(images,
                         np.array([[25, 20, 0],
                                   [-5, 0, 0]]))
        images = box.get_images(points)
        npt.assert_equal(images,
                         np.array([[25, 20, 0],
                                   [-5, 0, 0]]))

    def test_center_of_mass(self):
        box = freud.box.Box.cube(5)

        npt.assert_allclose(
            box.center_of_mass([[0, 0, 0]]), [0, 0, 0], atol=1e-6)
        npt.assert_allclose(
            box.center_of_mass([[1, 1, 1]]), [1, 1, 1], atol=1e-6)
        npt.assert_allclose(
            box.center_of_mass([[1, 1, 1], [2, 2, 2]]),
            [1.5, 1.5, 1.5], atol=1e-6)
        npt.assert_allclose(
            box.center_of_mass([[-2, -2, -2], [2, 2, 2]]),
            [-2.5, -2.5, -2.5], atol=1e-6)
        npt.assert_allclose(
            box.center_of_mass([[-2.2, -2.2, -2.2], [2, 2, 2]]),
            [2.4, 2.4, 2.4], atol=1e-6)

    def test_center_of_mass_weighted(self):
        box = freud.box.Box.cube(5)

        points = [[0, 0, 0], -box.L/4]
        masses = [2, 1]
        phases = np.exp(2*np.pi*1j*box.make_fractional(points))
        com_angle = np.angle(phases.T @ masses / np.sum(masses))
        com = box.make_absolute(com_angle / (2*np.pi))
        npt.assert_allclose(
            box.center_of_mass(points, masses), com, atol=1e-6)

    def test_center(self):
        box = freud.box.Box.cube(5)

        npt.assert_allclose(box.center([[0, 0, 0]]), [[0, 0, 0]], atol=1e-6)
        npt.assert_allclose(box.center([[1, 1, 1]]), [[0, 0, 0]], atol=1e-6)
        npt.assert_allclose(box.center([[1, 1, 1], [2, 2, 2]]),
                            [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], atol=1e-6)
        npt.assert_allclose(box.center([[-2, -2, -2], [2, 2, 2]]),
                            [[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]], atol=1e-6)

    def test_center_weighted(self):
        box = freud.box.Box.cube(5)

        points = [[0, 0, 0], -box.L/4]
        masses = [2, 1]
        phases = np.exp(2*np.pi*1j*box.make_fractional(points))
        com_angle = np.angle(phases.T @ masses / np.sum(masses))
        com = box.make_absolute(com_angle / (2*np.pi))
        npt.assert_allclose(box.center(points, masses),
                            box.wrap(points - com), atol=1e-6)

        # Make sure the center of mass is not (0, 0, 0) if ignoring masses
        assert not np.allclose(box.center_of_mass(points),
                               [0, 0, 0], atol=1e-6)

    def test_absolute_coordinates(self):
        box = freud.box.Box(2, 2, 2)
        f_point = np.array([[0.5, 0.25, 0.75],
                            [0, 0, 0],
                            [0.5, 0.5, 0.5]])
        point = np.array([[0, -0.5, 0.5], [-1, -1, -1], [0, 0, 0]])

        testcoordinates = np.array([box.make_absolute(f) for f in f_point])
        npt.assert_equal(testcoordinates, point)

        testcoordinates = box.make_absolute(f_point)

        npt.assert_equal(testcoordinates, point)

    def test_fractional_coordinates(self):
        box = freud.box.Box(2, 2, 2)
        f_point = np.array([[0.5, 0.25, 0.75],
                            [0, 0, 0],
                            [0.5, 0.5, 0.5]])
        point = np.array([[0, -0.5, 0.5], [-1, -1, -1], [0, 0, 0]])

        testfraction = np.array([box.make_fractional(vec) for vec in point])
        npt.assert_equal(testfraction, f_point)

        testfraction = box.make_fractional(point)

        npt.assert_equal(testfraction, f_point)

    def test_vectors(self):
        """Test getting lattice vectors"""
        b_list = [1, 2, 3, 0.1, 0.2, 0.3]
        Lx, Ly, Lz, xy, xz, yz = b_list
        box = freud.box.Box.from_box(b_list)
        npt.assert_allclose(
            box.get_box_vector(0),
            [Lx, 0, 0]
        )
        npt.assert_allclose(
            box.v1,
            [Lx, 0, 0]
        )
        npt.assert_allclose(
            box.get_box_vector(1),
            [xy*Ly, Ly, 0]
        )
        npt.assert_allclose(
            box.v2,
            [xy*Ly, Ly, 0]
        )
        npt.assert_allclose(
            box.get_box_vector(2),
            [xz*Lz, yz*Lz, Lz]
        )
        npt.assert_allclose(
            box.v3,
            [xz*Lz, yz*Lz, Lz]
        )

    def test_periodic(self):
        box = freud.box.Box(1, 2, 3, 0, 0, 0)
        npt.assert_array_equal(box.periodic, True)
        self.assertTrue(box.periodic_x)
        self.assertTrue(box.periodic_y)
        self.assertTrue(box.periodic_z)

        # Test setting all flags together
        box.periodic = False
        npt.assert_array_equal(box.periodic, False)
        self.assertFalse(box.periodic_x)
        self.assertFalse(box.periodic_y)
        self.assertFalse(box.periodic_z)

        # Test setting flags as a list
        box.periodic = [True, True, True]
        npt.assert_array_equal(box.periodic, True)

        # Test setting each flag separately
        box.periodic_x = False
        box.periodic_y = False
        box.periodic_z = False
        self.assertEqual(box.periodic_x, False)
        self.assertEqual(box.periodic_y, False)
        self.assertEqual(box.periodic_z, False)

        box.periodic = True
        npt.assert_array_equal(box.periodic, True)

    def test_equal(self):
        box = freud.box.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = freud.box.Box(2, 2, 2, 1, 0, 0)
        self.assertEqual(box, box)
        self.assertNotEqual(box, box2)

    def test_repr(self):
        box = freud.box.Box(2, 2, 2, 1, 0.5, 0.1)
        self.assertEqual(box, eval(repr(box)))

    def test_str(self):
        box = freud.box.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = freud.box.Box(2, 2, 2, 1, 0.5, 0.1)
        self.assertEqual(str(box), str(box2))

    def test_to_dict(self):
        """Test converting box to dict"""
        box = freud.box.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = box.to_dict()
        box_dict = {'Lx': 2, 'Ly': 2, 'Lz': 2, 'xy': 1, 'xz': 0.5, 'yz': 0.1}
        for k in box_dict:
            npt.assert_allclose(box_dict[k], box2[k])

    def test_from_box(self):
        """Test various methods of initializing a box"""
        box = freud.box.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = freud.box.Box.from_box(box)
        self.assertEqual(box, box2)

        box_dict = {'Lx': 2, 'Ly': 2, 'Lz': 2, 'xy': 1, 'xz': 0.5, 'yz': 0.1}
        box3 = freud.box.Box.from_box(box_dict)
        self.assertEqual(box, box3)

        with self.assertRaises(ValueError):
            box_dict['dimensions'] = 3
            freud.box.Box.from_box(box_dict, 2)

        BoxTuple = namedtuple('BoxTuple',
                              ['Lx', 'Ly', 'Lz', 'xy', 'xz', 'yz',
                               'dimensions'])
        box4 = freud.box.Box.from_box(BoxTuple(2, 2, 2, 1, 0.5, 0.1, 3))
        self.assertEqual(box, box4)

        with self.assertRaises(ValueError):
            freud.box.Box.from_box(BoxTuple(2, 2, 2, 1, 0.5, 0.1, 2), 3)

        box5 = freud.box.Box.from_box([2, 2, 2, 1, 0.5, 0.1])
        self.assertEqual(box, box5)

        box6 = freud.box.Box.from_box(np.array([2, 2, 2, 1, 0.5, 0.1]))
        self.assertEqual(box, box6)

        with self.assertRaises(ValueError):
            freud.box.Box.from_box([2, 2, 2, 1, 0.5])

        box7 = freud.box.Box.from_matrix(box.to_matrix())
        self.assertTrue(np.isclose(box.to_matrix(), box7.to_matrix()).all())

    def test_matrix(self):
        box = freud.box.Box(2, 2, 2, 1, 0.5, 0.1)
        box2 = freud.box.Box.from_matrix(box.to_matrix())
        self.assertTrue(np.isclose(box.to_matrix(), box2.to_matrix()).all())

        box3 = freud.box.Box(2, 2, 0, 0.5, 0, 0)
        box4 = freud.box.Box.from_matrix(box3.to_matrix())
        self.assertTrue(np.isclose(box3.to_matrix(), box4.to_matrix()).all())

    def test_set_dimensions(self):
        b = np.asarray([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        box = freud.Box.from_box(b, dimensions=2)
        self.assertTrue(box.dimensions == 2)

    def test_2_dimensional(self):
        box = freud.box.Box.square(L=1)
        # Setting Lz for a 2D box throws a warning that we hide with setUp()
        box.Lz = 1.0
        self.assertEqual(box.Lz, 0.0)
        self.assertTrue(box.dimensions == 2)
        box.dimensions = 3
        self.assertEqual(box.Lz, 0.0)
        self.assertTrue(box.dimensions == 3)
        box.Lz = 1.0
        self.assertEqual(box.Lz, 1.0)

    def test_cube(self):
        L = 10.0
        cube = freud.box.Box.cube(L=L)
        self.assertEqual(cube.Lx, L)
        self.assertEqual(cube.Ly, L)
        self.assertEqual(cube.Lz, L)
        self.assertEqual(cube.xy, 0)
        self.assertEqual(cube.xz, 0)
        self.assertEqual(cube.yz, 0)
        self.assertEqual(cube.dimensions, 3)

    def test_square(self):
        L = 10.0
        square = freud.box.Box.square(L=L)
        self.assertEqual(square.Lx, L)
        self.assertEqual(square.Ly, L)
        self.assertEqual(square.Lz, 0)
        self.assertEqual(square.xy, 0)
        self.assertEqual(square.xz, 0)
        self.assertEqual(square.yz, 0)
        self.assertEqual(square.dimensions, 2)

    def test_multiply(self):
        box = freud.box.Box(2, 3, 4, 1, 0.5, 0.1)
        box2 = box * 2
        self.assertTrue(np.isclose(box2.Lx, 4))
        self.assertTrue(np.isclose(box2.Ly, 6))
        self.assertTrue(np.isclose(box2.Lz, 8))
        self.assertTrue(np.isclose(box2.xy, 1))
        self.assertTrue(np.isclose(box2.xz, 0.5))
        self.assertTrue(np.isclose(box2.yz, 0.1))
        box3 = 2 * box
        self.assertEqual(box2, box3)

    def test_plot_3d(self):
        box = freud.box.Box(2, 3, 4, 1, 0.5, 0.1)
        box.plot()

    def test_plot_2d(self):
        box = freud.box.Box(2, 3, 0, 1, 0, 0, is2D=True)
        box.plot()

    def test_compute_distances_2d(self):
        box = freud.box.Box(2, 3, 0, 1, 0, 0, is2D=True)
        points = np.array([[0, 0, 0], [-2.2, -1.3, 0]])
        query_points = np.array(
            [[-0.5, -1.3, 0.], [0.5, 0, 0], [-2.2, -1.3, 0.], [0, 0.4, 0]])
        point_indices = np.array([1, 0, 1, 0])
        query_point_indices = np.array([0, 1, 2, 3])
        distances = box.compute_distances(
            query_points[query_point_indices], points[point_indices])
        npt.assert_allclose(distances, [0.3, 0.5, 0.0, 0.4], rtol=1e-6)

        # 1 dimensional array
        distances = box.compute_distances(query_points[0], points[1])
        npt.assert_allclose(distances, [0.3], rtol=1e-6)

        with self.assertRaises(ValueError):
            box.compute_distances(
                query_points[query_point_indices[:-1]], points[point_indices])
        with self.assertRaises(ValueError):
            box.compute_distances(
                query_points[query_point_indices], points[point_indices[:-1]])

    def test_compute_distances_3d(self):
        box = freud.box.Box(2, 3, 4, 1, 0, 0)
        points = np.array([[0, 0, 0], [-2.2, -1.3, 2]])
        query_points = np.array(
            [[-0.5, -1.3, 2.], [0.5, 0, 0], [-2.2, -1.3, 2.], [0, 0, 0.2]])
        point_indices = np.array([1, 0, 1, 0])
        query_point_indices = np.array([0, 1, 2, 3])
        distances = box.compute_distances(
            query_points[query_point_indices], points[point_indices])
        npt.assert_allclose(distances, [0.3, 0.5, 0.0, 0.2], rtol=1e-6)

    def test_compute_all_distances_2d(self):
        box = freud.box.Box(2, 3, 0, 1, 0, 0, is2D=True)
        points = np.array([[0., 0., 0.], [0., 0., 0.]])
        query_points = np.array(
            [[0.2, 0., 0.], [0., -0.4, 0.], [1., 1., 0.]])
        distances = box.compute_all_distances(points, query_points)
        npt.assert_allclose(distances, [
            [0.2, 0.4, np.sqrt(2)],
            [0.2, 0.4, np.sqrt(2)]], rtol=1e-6)

        points = np.array([0., 0., 0.])
        distances = box.compute_all_distances(points, query_points)
        npt.assert_allclose(distances, [[0.2, 0.4, np.sqrt(2)]], rtol=1e-6)

    def test_compute_all_distances_3d(self):
        box = freud.box.Box(2, 3, 4, 1, 0, 0)
        points = np.array([[0., 0., 1.], [0., 0., 0.]])
        query_points = np.array([[1., 0., 1.], [0., 0., 1.], [0., 0., 0.]])
        distances = box.compute_all_distances(points, query_points)
        npt.assert_allclose(distances,
                            [[1., 0., 1.], [np.sqrt(2), 1., 0.]], rtol=1e-6)

    def test_contains_2d(self):
        box = freud.box.Box(2, 3, 0, 1, 0, 0)
        points = np.random.uniform(-0.5, 0.5, size=(100, 3)).astype(np.float32)
        points[:50] = np.random.uniform(
            0.50001, 0.6, size=(50, 3)).astype(np.float32)
        points[:50] *= (-1)**np.random.randint(0, 2, size=(50, 3))
        points = points @ box.to_matrix().T
        # Force z=0
        points[:, 2] = 0

        in_box_mask = np.ones(points.shape[0]).astype(bool)
        in_box_mask[:50] = False
        npt.assert_array_equal(in_box_mask, box.contains(points))

    def test_contains_3d(self):
        box = freud.box.Box(2, 3, 4, 1, 0.1, 0.3)
        points = np.random.uniform(-0.5, 0.5, size=(100, 3)).astype(np.float32)
        points[:50] = np.random.uniform(
            0.50001, 0.6, size=(50, 3)).astype(np.float32)
        points[:50] *= (-1)**np.random.randint(0, 2, size=(50, 3))
        points = points @ box.to_matrix().T

        in_box_mask = np.ones(points.shape[0]).astype(bool)
        in_box_mask[:50] = False
        npt.assert_array_equal(in_box_mask, box.contains(points))


if __name__ == '__main__':
    unittest.main()
