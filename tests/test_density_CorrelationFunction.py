import numpy as np
import numpy.testing as npt
import freud
import unittest
import util


class TestCorrelationFunction(unittest.TestCase):
    def test_generateR(self):
        r_max = 51.23
        dr = 0.1
        nbins = int(r_max / dr)

        # make sure the radius for each bin is generated correctly
        r_list = np.zeros(nbins, dtype=np.float32)
        for i in range(nbins):
            r1 = i * dr
            r2 = r1 + dr
            r_list[i] = 2.0/3.0 * (r2**3.0 - r1**3.0) / (r2**2.0 - r1**2.0)

        ocf = freud.density.CorrelationFunction(r_max, dr)

        npt.assert_allclose(ocf.R, r_list, atol=1e-3)

    def test_attribute_access(self):
        r_max = 10.0
        dr = 1.0
        num_points = 100
        box_size = r_max*3.1
        box, points = util.make_box_and_random_points(
            box_size, num_points, True)
        ang = np.random.random_sample((num_points)).astype(np.float64) \
            * 2.0 * np.pi
        ocf = freud.density.CorrelationFunction(r_max, dr)

        # Test protected attribute access
        with self.assertRaises(AttributeError):
            ocf.RDF
        with self.assertRaises(AttributeError):
            ocf.box
        with self.assertRaises(AttributeError):
            ocf.counts

        ocf.accumulate(box, points, ang)

        # Test if accessible now
        ocf.RDF
        ocf.box
        ocf.counts

        # reset
        ocf.reset()

        # Test protected attribute access
        with self.assertRaises(AttributeError):
            ocf.RDF
        with self.assertRaises(AttributeError):
            ocf.box
        with self.assertRaises(AttributeError):
            ocf.counts

        ocf.compute(box, points, ang)

        # Test if accessible now
        ocf.RDF
        ocf.box
        ocf.counts

    def test_random_points_complex(self):
        r_max = 10.0
        dr = 1.0
        num_points = 1000
        box_size = r_max*3.1
        box, points = util.make_box_and_random_points(
            box_size, num_points, True)
        ang = np.random.random_sample((num_points)).astype(np.float64) \
            * 2.0 * np.pi
        comp = np.exp(1j*ang)
        correct = np.zeros(int(r_max/dr), dtype=np.complex64)
        absolute_tolerance = 0.1
        # first bin is bad
        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, 'ball', r_max, 0, True)
        for ts in test_set:
            ocf = freud.density.CorrelationFunction(r_max, dr)
            ocf.accumulate(box, ts[0], comp, points, np.conj(comp),
                           query_args={"r_max": r_max, "exclude_ii": True},
                           nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.compute(box, ts[0], comp, points, np.conj(comp),
                        query_args={"r_max": r_max, "exclude_ii": True},
                        nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            self.assertEqual(box, ocf.box)

            ocf.reset()
            ocf.accumulate(
                box, ts[0], comp, query_values=np.conj(comp), nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.compute(
                box, ts[0], comp, query_values=np.conj(comp), nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)

    def test_random_points_real(self):
        r_max = 10.0
        dr = 1.0
        num_points = 1000
        box_size = r_max*3.1
        box, points = util.make_box_and_random_points(
            box_size, num_points, True)
        ang = np.random.random_sample((num_points)).astype(np.float64) - 0.5
        correct = np.zeros(int(r_max/dr), dtype=np.float64)
        absolute_tolerance = 0.1
        # first bin is bad
        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, 'ball', r_max, 0, True)
        for ts in test_set:
            ocf = freud.density.CorrelationFunction(r_max, dr)
            ocf.accumulate(box, ts[0], ang, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.compute(box, ts[0], ang, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.reset()
            ocf.accumulate(box, ts[0], ang, points, ang,
                           query_args={'r_max': r_max, 'exclude_ii': True},
                           nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.reset()
            ocf.accumulate(box, ts[0], ang, query_values=ang, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            ocf.compute(box, ts[0], ang, nlist=ts[1])
            npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)
            self.assertEqual(freud.box.Box.square(box_size), ocf.box)

    def test_zero_points_complex(self):
        r_max = 10.0
        dr = 1.0
        num_points = 1000
        box_size = r_max*3.1
        box, points = util.make_box_and_random_points(
            box_size, num_points, True)
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.exp(1j*ang)
        ocf = freud.density.CorrelationFunction(r_max, dr)
        ocf.accumulate(freud.box.Box.square(box_size), points, comp,
                       points, np.conj(comp),
                       query_args={"r_max": r_max, "exclude_ii": True})

        correct = np.ones(int(r_max/dr), dtype=np.float32) + \
            1j * np.zeros(int(r_max/dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)

    def test_zero_points_real(self):
        r_max = 10.0
        dr = 1.0
        num_points = 1000
        box_size = r_max*3.1
        box, points = util.make_box_and_random_points(
            box_size, num_points, True)
        ang = np.zeros(int(num_points), dtype=np.float64)
        ocf = freud.density.CorrelationFunction(r_max, dr)
        ocf.accumulate(box, points, ang)

        correct = np.zeros(int(r_max/dr), dtype=np.float32)
        absolute_tolerance = 0.1
        npt.assert_allclose(ocf.RDF, correct, atol=absolute_tolerance)

    def test_counts(self):
        r_max = 10.0
        dr = 1.0
        num_points = 10
        box_size = r_max*2.1
        box, points = util.make_box_and_random_points(
            box_size, num_points, True)
        ang = np.zeros(int(num_points), dtype=np.float64)
        comp = np.exp(1j*ang)

        vectors = points[np.newaxis, :, :] - points[:, np.newaxis, :]
        vector_lengths = np.array(
            [[np.linalg.norm(box.wrap(vectors[i][j]))
              for i in range(num_points)]
             for j in range(num_points)])

        # Subtract len(points) to exclude the zero i-i distances
        correct = np.sum(vector_lengths < r_max) - len(points)
        ocf = freud.density.CorrelationFunction(r_max, dr)
        ocf.compute(freud.box.Box.square(box_size), points, comp,
                    points, np.conj(comp),
                    query_args={"r_max": r_max, "exclude_ii": True})
        self.assertEqual(np.sum(ocf.counts), correct)

    @unittest.skip('Skipping slow summation test.')
    def test_summation(self):
        # Causes the correlation function to add lots of small numbers together
        # This leads to vastly different results with different numbers of
        # threads if the summation is not done robustly
        N = 20000
        L = 1000
        phi = np.random.rand(N)
        box, pos2d = util.make_box_and_random_points(L, N, True)

        # With a small number of particles, we won't get the average exactly
        # right, so we check for different behavior with different numbers of
        # threads
        freud.parallel.setNumThreads(1)
        # A very large bin size exacerbates the problem
        cf = freud.density.CorrelationFunction(L/2.1, 40)
        cf.compute(box, pos2d, phi)
        c1 = cf.counts
        f1 = np.real(cf.RDF)

        freud.parallel.setNumThreads(20)
        cf.compute(box, pos2d, phi)
        c2 = cf.counts
        f2 = np.real(cf.RDF)

        npt.assert_allclose(f1, f2)
        npt.assert_array_equal(c1, c2)

    def test_repr(self):
        cf = freud.density.CorrelationFunction(1000, 40)
        self.assertEqual(str(cf), str(eval(repr(cf))))

    def test_repr_png(self):
        r_max = 10.0
        dr = 1.0
        num_points = 100
        box_size = r_max*3.1
        box, points = util.make_box_and_random_points(
            box_size, num_points, True)
        ang = np.random.random_sample((num_points)).astype(np.float64) \
            * 2.0 * np.pi
        comp = np.exp(1j*ang)
        ocf = freud.density.CorrelationFunction(r_max, dr)

        with self.assertRaises(AttributeError):
            ocf.plot()
        self.assertEqual(ocf._repr_png_(), None)

        ocf.accumulate(freud.box.Box.square(box_size), points, comp,
                       points, np.conj(comp),
                       query_args={"r_max": r_max, "exclude_ii": True})
        ocf._repr_png_()

    def test_query_nn_complex(self):
        """Test nearest-neighbor-based querying."""
        box_size = 8
        r_max = 3
        dr = 1
        box = freud.box.Box.cube(box_size)
        ref_points = np.array([[0, 0, 0]],
                              dtype=np.float32)
        points = np.array([[0.4, 0.0, 0.0],
                           [0.9, 0.0, 0.0],
                           [0.0, 1.4, 0.0],
                           [0.0, 1.9, 0.0],
                           [0.0, 0.0, 2.4],
                           [0.0, 0.0, 2.9]],
                          dtype=np.float32)
        ref_values = np.ones(ref_points.shape[0])
        values = np.ones(points.shape[0])

        cf = freud.density.CorrelationFunction(r_max, dr)
        cf.compute(box, ref_points, ref_values, points, values,
                   query_args={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(cf.RDF, [1, 1, 1])
        npt.assert_array_equal(cf.counts, [2, 2, 2])

        cf.compute(box, points, values, ref_points, ref_values,
                   query_args={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(cf.RDF, [1, 0, 0])
        npt.assert_array_equal(cf.counts, [1, 0, 0])

        ref_values = [1+1j]
        values = [1+1j, 1+1j, 2+2j, 2+2j, 3+3j, 3+3j]
        cf.compute(box, ref_points, ref_values, points, np.conj(values),
                   query_args={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(cf.RDF, [2, 4, 6])
        npt.assert_array_equal(cf.counts, [2, 2, 2])

        cf.compute(box, ref_points, ref_values, points, values,
                   query_args={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(cf.RDF, [2j, 4j, 6j])
        npt.assert_array_equal(cf.counts, [2, 2, 2])

        cf.compute(box, points, values, ref_points, np.conj(ref_values),
                   query_args={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(cf.RDF, [2, 0, 0])
        npt.assert_array_equal(cf.counts, [1, 0, 0])

        cf.compute(box, points, values, ref_points, ref_values,
                   query_args={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(cf.RDF, [2j, 0, 0])
        npt.assert_array_equal(cf.counts, [1, 0, 0])

    def test_query_nn_real(self):
        """Test nearest-neighbor-based querying."""
        box_size = 8
        r_max = 3
        dr = 1
        box = freud.box.Box.cube(box_size)
        ref_points = np.array([[0, 0, 0]],
                              dtype=np.float32)
        points = np.array([[0.4, 0.0, 0.0],
                           [0.9, 0.0, 0.0],
                           [0.0, 1.4, 0.0],
                           [0.0, 1.9, 0.0],
                           [0.0, 0.0, 2.4],
                           [0.0, 0.0, 2.9]],
                          dtype=np.float32)
        ref_values = np.ones(ref_points.shape[0])
        values = np.ones(points.shape[0])

        cf = freud.density.CorrelationFunction(r_max, dr)
        cf.compute(box, ref_points, ref_values, points, values,
                   query_args={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(cf.RDF, [1, 1, 1])
        npt.assert_array_equal(cf.counts, [2, 2, 2])

        cf.compute(box, points, values, ref_points, ref_values,
                   query_args={'mode': 'nearest', 'num_neighbors': 1})
        npt.assert_array_equal(cf.RDF, [1, 0, 0])
        npt.assert_array_equal(cf.counts, [1, 0, 0])

    def test_points_ne_query_points_complex(self):
        # Helper function to give complex number representation of a point
        def value_func(_p):
            return _p[0] + 1j*_p[1]

        r_max = 10.0
        dr = 0.1
        box_size = r_max*5
        box = freud.box.Box.square(box_size)

        ocf = freud.density.CorrelationFunction(r_max, dr)

        query_points = []
        query_values = []
        N = 300

        # We are essentially generating all n-th roots of unity
        # scalar multiplied by the each bin centers
        # with the value of a point being its complex number representation.
        # Therefore, the RDF should be uniformly zero
        # since the roots of unity add up to zero, if we set our ref_point in
        # the origin.
        # Nice proof for this fact is that when the n-th roots of unity
        # are viewed as vectors, we can draw a regular n-gon
        # so that we start at the origin and come back to origin.
        for r in ocf.R:
            for k in range(N):
                point = [r * np.cos(2*np.pi*k/N), r * np.sin(2*np.pi*k/N), 0]
                query_points.append(point)
                query_values.append(value_func(point))

        supposed_RDF = np.zeros(ocf.R.shape)

        # points are within distances closer than dr, so their impact on
        # the result should be minimal.
        points = [[dr/4, 0, 0], [-dr/4, 0, 0], [0, dr/4, 0], [0, -dr/4, 0]]

        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, "ball", r_max, 0, False)
        for ts in test_set:
            ocf = freud.density.CorrelationFunction(r_max, dr)
            # try for different scalar values.
            for rv in [0, 1, 2, 7]:
                values = [rv] * 4

                ocf.compute(
                    box, ts[0], values,
                    query_points, query_values, nlist=ts[1])
                correct = supposed_RDF

                npt.assert_allclose(ocf.RDF, correct, atol=1e-6)

    def test_points_ne_query_points_real(self):
        def value_func(_r):
            return np.sin(_r)

        r_max = 10.0
        dr = 0.1
        box_size = r_max*5
        box = freud.box.Box.square(box_size)

        ocf = freud.density.CorrelationFunction(r_max, dr)

        query_points = []
        query_values = []
        supposed_RDF = []
        N = 300

        # We are generating the values so that they are sine wave from 0 to 2pi
        # rotated around z axis.
        # Therefore, the RDF should be a scalar multiple sin if we set our
        # ref_point to be in the origin.
        for r in ocf.R:
            for k in range(N):
                query_points.append([r * np.cos(2*np.pi*k/N),
                                     r * np.sin(2*np.pi*k/N), 0])
                query_values.append(value_func(r))
            supposed_RDF.append(value_func(r))

        supposed_RDF = np.array(supposed_RDF)

        # points are within distances closer than dr, so their impact on
        # the result should be minimal.
        points = [[dr/4, 0, 0], [-dr/4, 0, 0], [0, dr/4, 0], [0, -dr/4, 0]]

        test_set = util.make_raw_query_nlist_test_set(
            box, points, query_points, "ball", r_max, 0, False)
        for ts in test_set:
            ocf = freud.density.CorrelationFunction(r_max, dr)
            # try for different scalar values.
            for rv in [0, 1, 2, 7]:
                values = [rv] * 4

                ocf.compute(
                    box, ts[0], values,
                    query_points, query_values, nlist=ts[1])
                correct = supposed_RDF * rv

                npt.assert_allclose(ocf.RDF, correct, atol=1e-6)


if __name__ == '__main__':
    unittest.main()