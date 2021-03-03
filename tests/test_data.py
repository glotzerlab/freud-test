import freud
import unittest
import numpy.testing as npt
import numpy as np
from util import sort_rounded_xyz_array


class TestUnitCell(unittest.TestCase):
    def test_square(self):
        """Test that the square lattice is correctly generated."""
        box, points = freud.data.UnitCell.square().generate_system()
        self.assertEqual(box, freud.box.Box.square(1))
        npt.assert_array_equal(points, [[-0.5, -0.5, 0]])

    def test_sc(self):
        """Test that the sc lattice is correctly generated."""
        box, points = freud.data.UnitCell.sc().generate_system()
        self.assertEqual(box, freud.box.Box.cube(1))
        npt.assert_array_equal(points, [[-0.5, -0.5, -0.5]])

    def test_bcc(self):
        """Test that the bcc lattice is correctly generated."""
        box, points = freud.data.UnitCell.bcc().generate_system()
        self.assertEqual(box, freud.box.Box.cube(1))
        npt.assert_array_equal(points, [[0, 0, 0],
                                        [-0.5, -0.5, -0.5]])

    def test_fcc(self):
        """Test that the fcc lattice is correctly generated."""
        box, points = freud.data.UnitCell.fcc().generate_system()
        self.assertEqual(box, freud.box.Box.cube(1))
        npt.assert_array_equal(points,
                               [[0, 0, -0.5],
                                [0, -0.5, 0],
                                [-0.5, 0, 0],
                                [-0.5, -0.5, -0.5]])

    def test_scale(self):
        """Test the generation of a scaled structure."""
        for scale in [0.5, 2]:
            box, points = freud.data.UnitCell.fcc().generate_system(
                scale=scale)
            self.assertEqual(box, freud.box.Box.cube(scale))
            npt.assert_array_equal(
                points,
                scale*np.array([[0, 0, -0.5],
                                [0, -0.5, 0],
                                [-0.5, 0, 0],
                                [-0.5, -0.5, -0.5]]))

    def test_replicas(self):
        """Test that replication works."""
        for num_replicas in range(1, 10):
            box, points = freud.data.UnitCell.fcc().generate_system(
                num_replicas=num_replicas)
            self.assertEqual(box, freud.box.Box.cube(num_replicas))

            test_points = np.array([[0, 0.5, 0.5],
                                    [0.5, 0, 0.5],
                                    [0.5, 0.5, 0],
                                    [0., 0., 0.]])
            test_points = test_points[np.newaxis, np.newaxis, np.newaxis, ...]
            test_points = np.tile(
                test_points, [num_replicas, num_replicas, num_replicas, 1, 1])
            test_points[..., 0] += np.arange(
                num_replicas)[:, np.newaxis, np.newaxis, np.newaxis]
            test_points[..., 1] += np.arange(
                num_replicas)[np.newaxis, :, np.newaxis, np.newaxis]
            test_points[..., 2] += np.arange(
                num_replicas)[np.newaxis, np.newaxis, :, np.newaxis]
            test_points = (test_points-(num_replicas*0.5)).reshape(-1, 3)

            npt.assert_allclose(
                sort_rounded_xyz_array(points),
                sort_rounded_xyz_array(box.wrap(test_points))
            )

    def test_invalid_replicas(self):
        """Test that invalid replications raise errors."""
        for num_replicas in (0, 2.5, -1, [2, 2, 0], [2, 2, 2], 'abc'):
            with self.assertRaises(ValueError):
                freud.data.UnitCell.square().generate_system(
                    num_replicas=num_replicas)

    def test_noise(self):
        """Test that noise generation works."""
        sigma = 0.01
        box, points = freud.data.UnitCell.fcc().generate_system(
            sigma_noise=sigma, seed=0)
        self.assertEqual(box, freud.box.Box.cube(1))

        test_points = np.array([[0, 0, -0.5],
                                [0, -0.5, 0],
                                [-0.5, 0, 0],
                                [-0.5, -0.5, -0.5]])

        deltas = np.linalg.norm(box.wrap(test_points-points), axis=-1)
        # Nothing should be exactly equal, but differences should not be too
        # large. 4 sigma is an arbitrary choice that gives a high probability
        # of the test passing (although not necessary since the seed is set
        # above).
        self.assertFalse(np.allclose(deltas, 0))
        npt.assert_allclose(deltas, 0, atol=4*sigma)

    def test_seed(self):
        """Ensure that seeding does not overwrite the global random state."""
        num_points = 10
        sigma = 0.01

        np.random.seed(0)
        first_rand = np.random.randint(num_points)
        box, points = freud.data.UnitCell.fcc().generate_system(
            sigma_noise=sigma, seed=1)
        second_rand = np.random.randint(num_points)

        np.random.seed(0)
        third_rand = np.random.randint(num_points)
        box, points = freud.data.UnitCell.fcc().generate_system(
            sigma_noise=sigma, seed=2)
        fourth_rand = np.random.randint(num_points)

        npt.assert_array_equal(first_rand, third_rand)
        npt.assert_array_equal(second_rand, fourth_rand)


class TestRandomSystem(unittest.TestCase):
    def test_sizes_and_dimensions(self):
        for N in (0, 1, 10, 100, 1000):
            for is2D in (True, False):
                box, points = freud.data.make_random_system(
                    box_size=10, num_points=N, is2D=is2D)
                self.assertEqual(points.shape, (N, 3))
                self.assertEqual(box.is2D, is2D)

    def test_seed(self):
        """Ensure that seeding does not overwrite the global random state."""
        box_size = 10
        num_points = 10

        np.random.seed(0)
        first_rand = np.random.randint(num_points)
        box, points = freud.data.make_random_system(
            box_size=box_size, num_points=num_points, seed=1)
        second_rand = np.random.randint(num_points)

        np.random.seed(0)
        third_rand = np.random.randint(num_points)
        box, points = freud.data.make_random_system(
            box_size=box_size, num_points=num_points, seed=2)
        fourth_rand = np.random.randint(num_points)

        npt.assert_array_equal(first_rand, third_rand)
        npt.assert_array_equal(second_rand, fourth_rand)


if __name__ == '__main__':
    unittest.main()
