import unittest

import matplotlib
import numpy as np
import numpy.testing as npt

import freud

matplotlib.use("agg")


def validate_method(system, bins, k_max, k_min):
    """Validation of the static structure calculation.

    This method is a pure Python reference implementation of the debye
    method implemented in C++ in freud.

    Args:
        system:
            Any object that is a valid argument to
            :class:`freud.locality.NeighborQuery.from_system`.
        bins (unsigned int):
            Number of bins in :math:`k` space.
        k_max (float):
            Maximum :math:`k` value to include in the calculation.
        k_min (float):
            Minimum :math:`k` value to include in the calculation.
    """
    rmax = np.min(system[0].L) * 0.5 - 0.0000001
    system = freud.locality.NeighborQuery.from_system(system)
    N = len(system.points)

    Q = np.linspace(k_min, k_max, bins, endpoint=False)
    Q += (k_max - k_min) / bins / 2
    S = np.zeros_like(Q)

    # Compute all pairwise distances
    query_args = dict(mode="ball", r_max=rmax, exclude_ii=True)
    distances = system.query(system.points, query_args).toNeighborList().distances

    for i, q in enumerate(Q):
        S[i] += np.sum(np.sinc(q * distances / np.pi)) / N

    return Q, S


class TestStaticStructureFactorDebye(unittest.TestCase):
    def test_compute(self):
        sf = freud.diffraction.StaticStructureFactorDebye(1000, 100)
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        sf.compute((box, positions))

    def test_debye_validation(self):
        """Validate the Debye method against a Python implementation."""
        bins = 1000
        k_max = 100
        k_min = 0
        sf = freud.diffraction.StaticStructureFactorDebye(bins, k_max, k_min)
        box, positions = freud.data.UnitCell.fcc().generate_system(4, sigma_noise=0.01)
        sf.compute((box, positions))
        Q, S = validate_method((box, positions), bins, k_max, k_min)
        npt.assert_allclose(sf.bin_centers, Q)
        npt.assert_allclose(sf.S_k, S, rtol=1e-5, atol=1e-5)


# TODO: All the below tests were copied from DiffractionPattern and need to be
# updated for this class

#    def test_attribute_access(self):
#        grid_size = 234
#        output_size = 123
#        sf = freud.diffraction.StaticStructureFactorDebye(1000, 100)
#        self.assertEqual(sf.grid_size, grid_size)
#        self.assertEqual(sf.output_size, grid_size)
#        sf = freud.diffraction.StaticStructureFactorDebye(
#            grid_size=grid_size, output_size=output_size)
#        self.assertEqual(sf.grid_size, grid_size)
#        self.assertEqual(sf.output_size, output_size)
#
#        box, positions = freud.data.UnitCell.fcc().generate_system(4)
#
#        with self.assertRaises(AttributeError):
#            sf.diffraction
#        with self.assertRaises(AttributeError):
#            sf.k_values
#        with self.assertRaises(AttributeError):
#            sf.k_vectors
#        with self.assertRaises(AttributeError):
#            sf.plot()
#
#        sf.compute((box, positions), zoom=1, peak_width=4)
#        diff = sf.diffraction
#        vals = sf.k_values
#        vecs = sf.k_vectors
#        sf.plot()
#        sf._repr_png_()
#
#        # Make sure old data is not invalidated by new call to compute()
#        box2, positions2 = freud.data.UnitCell.bcc().generate_system(3)
#        sf.compute((box2, positions2), zoom=1, peak_width=4)
#        self.assertFalse(np.array_equal(sf.diffraction, diff))
#        self.assertFalse(np.array_equal(sf.k_values, vals))
#        self.assertFalse(np.array_equal(sf.k_vectors, vecs))
#
#    def test_attribute_shapes(self):
#        grid_size = 234
#        output_size = 123
#        sf = freud.diffraction.StaticStructureFactorDebye(
#            grid_size=grid_size, output_size=output_size)
#        box, positions = freud.data.UnitCell.fcc().generate_system(4)
#        sf.compute((box, positions))
#
#        self.assertEqual(sf.diffraction.shape, (output_size, output_size))
#        self.assertEqual(sf.k_values.shape, (output_size,))
#        self.assertEqual(sf.k_vectors.shape, (output_size, output_size, 3))
#        self.assertEqual(sf.to_image().shape, (output_size, output_size, 4))
#
#    def test_repr(self):
#        sf = freud.diffraction.StaticStructureFactorDebye()
#        self.assertEqual(str(sf), str(eval(repr(sf))))
#
#        # Use non-default arguments for all parameters
#        sf = freud.diffraction.StaticStructureFactorDebye(
#            grid_size=123, output_size=234)
#        self.assertEqual(str(sf), str(eval(repr(sf))))
#
#    def test_k_values_and_k_vectors(self):
#        sf = freud.diffraction.StaticStructureFactorDebye()
#
#        for size in [2, 5, 10]:
#            for npoints in [10, 20, 75]:
#                box, positions = freud.data.make_random_system(npoints, size)
#                sf.compute((box, positions))
#
#                output_size = sf.output_size
#                npt.assert_allclose(sf.k_values[output_size//2], 0)
#                center_index = (output_size//2, output_size//2)
#                npt.assert_allclose(sf.k_vectors[center_index], [0, 0, 0])

if __name__ == "__main__":
    unittest.main()
