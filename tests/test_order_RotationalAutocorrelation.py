import unittest
import os
import math
import rowan

import freud

import numpy as np
import numpy.testing as npt


class TestRotationalAutocorrelation(unittest.TestCase):
    """Test the rotational autocorrelation order parameter"""
    def test_equality(self):
        """Ensure that autocorrelation against identical values is 1"""
        np.random.seed(24)
        orientations = np.random.rand(4, 4)
        orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]

        ra = freud.order.RotationalAutocorrelation(2)
        ra.compute(orientations, orientations)

        npt.assert_allclose(ra.order, 1, rtol=1e-6)
        npt.assert_allclose(ra.particle_order, 1, rtol=1e-6)

    def test_attributes(self):
        """Check that all attributes are sensible."""
        np.random.seed(24)
        orientations = np.random.rand(4, 4)
        orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]

        ra = freud.order.RotationalAutocorrelation(2)

        # Test access
        with self.assertRaises(AttributeError):
            ra.particle_order
        with self.assertRaises(AttributeError):
            ra.order

        ra.compute(orientations, orientations)

        # Test access
        ra.particle_order
        ra.order

        self.assertEqual(ra.l, 2)

    def test_data(self):
        """Regression test against known outputs."""
        fn = os.path.join(os.path.dirname(__file__),
                          'numpy_test_files',
                          'rotational_autocorrelation_orientations.npz')

        with np.load(fn) as data:
            orientations = data['orientations']

            ra2 = freud.order.RotationalAutocorrelation(2)
            l2 = []
            for i in range(orientations.shape[0]):
                ra2.compute(orientations[0, :, :], orientations[i, :, :])
                l2.append(ra2.order)
            npt.assert_allclose(l2, data['l2auto'], atol=1e-6, rtol=1e-6)

            ra6 = freud.order.RotationalAutocorrelation(6)
            l6 = []
            for i in range(orientations.shape[0]):
                ra6.compute(orientations[0, :, :], orientations[i, :, :])
                l6.append(ra6.order)
            npt.assert_allclose(l6, data['l6auto'], atol=1e-6, rtol=1e-6)

        # As a sanity check, make sure computing with the same object works on
        # new data.
        np.random.seed(42)
        orientations = np.random.rand(4, 4)
        orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]

        npt.assert_allclose(
            ra2.compute(orientations, orientations).order,
            1, rtol=1e-6)
        npt.assert_allclose(
            ra6.compute(orientations, orientations).order,
            1, rtol=1e-6)

    def test_repr(self):
        ra2 = freud.order.RotationalAutocorrelation(2)
        self.assertEqual(str(ra2), str(eval(repr(ra2))))


def quat_to_greek(q):
    '''Converts an array of quaternions to xi and zeta form.'''
    angle_array = np.zeros(shape=(len(q), 2), dtype=complex)
    angle_array[:, 0] = q[:, 1] + q[:, 2]*1j
    angle_array[:, 1] = q[:, 3] + q[:, 0]*1j
    return angle_array


def hypersphere_harmonic(angle_array, l, m1, m2):
    '''Calculates a single hyperspherical harmonical.'''
    a = - (m1 - l/2)
    b = - (m2 - l/2)

    val = 0
    if (l-a) >= 0 and (l-b) >= 0 and a >= 0 and b >= 0:
        for k in np.arange(0, 4*l+1):
            if (b-k >= 0) and (a-k >= 0) and (l + k - a - b >= 0):
                denom = (math.factorial(int(k)) *
                         math.factorial(int(l+k-a-b)) *
                         math.factorial(int(a-k)) *
                         math.factorial(int(b-k)))

                for xi, zeta in angle_array:
                    val += (
                        (xi.conjugate())**k *
                        (zeta)**(b-k) *
                        (zeta.conjugate())**(a-k) *
                        (-xi)**(l+k-a-b)) / denom

        val *= np.sqrt(
            (math.factorial(int(a)) *
             math.factorial(int(l-a)) *
             math.factorial(int(b)) *
             math.factorial(int(l-b))) / (l+1))

    return val/len(angle_array)


def return_correlation(l, initial_q, orientations):
    """Compute the rotational autocorrelation."""
    calc_quats = rowan.multiply(rowan.conjugate(initial_q), orientations)
    ref_quats = rowan.multiply(rowan.conjugate(initial_q), initial_q)

    ref_angles = quat_to_greek(ref_quats)
    calc_angles = quat_to_greek(calc_quats)

    f_of_t = 0

    for m1 in np.arange(-l/2, l/2+1):
        for m2 in np.arange(-l/2, l/2 + 1):
            ref_y = hypersphere_harmonic(ref_angles, l, m1, m2)
            calc_y = hypersphere_harmonic(calc_angles, l, m1, m2)
            f_of_t += (ref_y.conjugate() * calc_y)

    return f_of_t.real


class TestRotationalAutocorrelationReference(unittest.TestCase):
    """Test against a reference Python implementation."""
    N = 100
    for seed in range(5):
        for l in [4, 6, 8]:
            np.random.seed(seed)
            orientations = rowan.random.rand(N)
            ref_orientations = rowan.random.rand(N)

            ra = freud.order.RotationalAutocorrelation(l)
            ra.compute(ref_orientations, orientations)

            npt.assert_allclose(
                ra.order,
                return_correlation(l, ref_orientations, orientations),
                atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
