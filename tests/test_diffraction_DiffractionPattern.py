import matplotlib
import numpy as np
import numpy.testing as npt
import pytest
import rowan

import freud

matplotlib.use("agg")


class TestDiffractionPattern:
    def test_compute(self):
        dp = freud.diffraction.DiffractionPattern()
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        dp.compute((box, positions))

    @pytest.mark.parametrize("reset", [True, False])
    def test_reset(self, reset):
        dp_check = freud.diffraction.DiffractionPattern()
        dp_reference = freud.diffraction.DiffractionPattern()
        fcc = freud.data.UnitCell.fcc()
        box, positions = fcc.generate_system(4)

        for seed in range(2):
            # The check instance computes twice without resetting
            box, positions = fcc.generate_system(sigma_noise=1e-3, seed=seed)
            dp_check.compute((box, positions), reset=reset)

        # The reference instance computes once on the last system
        dp_reference.compute((box, positions))

        # If reset, then the data should be close. If not reset, the data
        # should be different.
        assert reset == np.allclose(dp_check.diffraction, dp_reference.diffraction)

    def test_attribute_access(self):
        grid_size = 234
        output_size = 123
        dp = freud.diffraction.DiffractionPattern(grid_size=grid_size)
        assert dp.grid_size == grid_size
        assert dp.output_size == grid_size
        dp = freud.diffraction.DiffractionPattern(
            grid_size=grid_size, output_size=output_size
        )
        assert dp.grid_size == grid_size
        assert dp.output_size == output_size

        box, positions = freud.data.UnitCell.fcc().generate_system(4)

        with pytest.raises(AttributeError):
            dp.diffraction
        with pytest.raises(AttributeError):
            dp.k_values
        with pytest.raises(AttributeError):
            dp.k_vectors
        with pytest.raises(AttributeError):
            dp.plot()

        dp.compute((box, positions), zoom=1, peak_width=4)
        diff = dp.diffraction
        vals = dp.k_values
        vecs = dp.k_vectors
        dp.plot()
        dp._repr_png_()

        # Make sure old data is not invalidated by new call to compute()
        box2, positions2 = freud.data.UnitCell.bcc().generate_system(3)
        dp.compute((box2, positions2), zoom=1, peak_width=4)
        assert not np.array_equal(dp.diffraction, diff)
        assert not np.array_equal(dp.k_values, vals)
        assert not np.array_equal(dp.k_vectors, vecs)

    def test_attribute_shapes(self):
        grid_size = 234
        output_size = 123
        dp = freud.diffraction.DiffractionPattern(
            grid_size=grid_size, output_size=output_size
        )
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        dp.compute((box, positions))

        assert dp.diffraction.shape == (output_size, output_size)
        assert dp.k_values.shape == (output_size,)
        assert dp.k_vectors.shape == (output_size, output_size, 3)
        assert dp.to_image().shape == (output_size, output_size, 4)

    def test_center_unordered(self):
        """Assert the center of the image is an intensity peak for an
        unordered system.
        """
        box, positions = freud.data.make_random_system(box_size=10, num_points=1000)

        # Test different parities (odd/even) of grid_size and output_size
        for grid_size in [255, 256]:
            for output_size in [255, 256]:
                dp = freud.diffraction.DiffractionPattern(
                    grid_size=grid_size, output_size=output_size
                )

                # Use a random view orientation and a random zoom
                for view_orientation in rowan.random.rand(10):
                    zoom = 1 + 10 * np.random.rand()
                    dp.compute(
                        system=(box, positions),
                        view_orientation=view_orientation,
                        zoom=zoom,
                    )

                    # Assert the pixel at the center (k=0) is the maximum value
                    diff = dp.diffraction
                    max_index = np.unravel_index(np.argmax(diff), diff.shape)
                    center_index = (output_size // 2, output_size // 2)
                    assert max_index == center_index

                    # The value at k=0 should be 1 because of normalization
                    # by (number of points)**2
                    npt.assert_allclose(dp.diffraction[center_index], 1)

    def test_center_ordered(self):
        """Assert the center of the image is an intensity peak for an ordered
        system.
        """
        box, positions = freud.data.UnitCell.bcc().generate_system(10)

        # Test different parities (odd/even) of grid_size and output_size
        for grid_size in [255, 256]:
            for output_size in [255, 256]:
                dp = freud.diffraction.DiffractionPattern(
                    grid_size=grid_size, output_size=output_size
                )
                # Use a random view orientation and a random zoom
                for view_orientation in rowan.random.rand(10):
                    zoom = 1 + 10 * np.random.rand()
                    dp.compute(
                        system=(box, positions),
                        view_orientation=view_orientation,
                        zoom=zoom,
                    )

                    # Assert the pixel at the center (k=0) is the maximum value
                    diff = dp.diffraction
                    max_index = np.unravel_index(np.argmax(diff), diff.shape)
                    center_index = (output_size // 2, output_size // 2)
                    assert max_index == center_index

                    # The value at k=0 should be 1 because of normalization
                    # by (number of points)**2
                    npt.assert_allclose(dp.diffraction[center_index], 1)

    def test_repr(self):
        dp = freud.diffraction.DiffractionPattern()
        assert str(dp) == str(eval(repr(dp)))

        # Use non-default arguments for all parameters
        dp = freud.diffraction.DiffractionPattern(grid_size=123, output_size=234)
        assert str(dp) == str(eval(repr(dp)))

    def test_k_values_and_k_vectors(self):
        dp = freud.diffraction.DiffractionPattern()

        for size in [2, 5, 10]:
            box, positions = freud.data.make_random_system(size, 1)
            zoom = 4
            view_orientation = np.asarray([1, 0, 0, 0])
            dp.compute((box, positions), view_orientation=view_orientation, zoom=zoom)

            output_size = dp.output_size
            npt.assert_allclose(dp.k_values[output_size // 2], 0)
            center_index = (output_size // 2, output_size // 2)
            npt.assert_allclose(dp.k_vectors[center_index], [0, 0, 0])

            # Tests for the first and last k value bins. Uses the math for
            # np.fft.fftfreq and the _k_scale_factor to write an expression for
            # the first and last bins' k values and k vectors.

            scale_factor = 2 * np.pi * output_size / (np.max(box.to_matrix()) * zoom)

            if output_size % 2 == 0:
                first_k_value = -0.5 * scale_factor
                last_k_value = (0.5 - (1 / output_size)) * scale_factor
            else:
                first_k_value = (-0.5 + 1 / (2 * output_size)) * scale_factor
                last_k_value = (0.5 - 1 / (2 * output_size)) * scale_factor

            npt.assert_allclose(dp.k_values[0], first_k_value)
            npt.assert_allclose(dp.k_values[-1], last_k_value)

            first_k_vector = rowan.rotate(
                view_orientation, [first_k_value, first_k_value, 0]
            )
            last_k_vector = rowan.rotate(
                view_orientation, [last_k_value, last_k_value, 0]
            )
            top_right_k_vector = rowan.rotate(
                view_orientation, [first_k_value, last_k_value, 0]
            )
            bottom_left_k_vector = rowan.rotate(
                view_orientation, [last_k_value, first_k_value, 0]
            )

            npt.assert_allclose(dp.k_vectors[0, 0], first_k_vector)
            npt.assert_allclose(dp.k_vectors[-1, -1], last_k_vector)
            npt.assert_allclose(dp.k_vectors[0, -1], top_right_k_vector)
            npt.assert_allclose(dp.k_vectors[-1, 0], bottom_left_k_vector)

            center = output_size // 2
            top_center_k_vector = rowan.rotate(view_orientation, [0, first_k_value, 0])
            bottom_center_k_vector = rowan.rotate(
                view_orientation, [0, last_k_value, 0]
            )
            left_center_k_vector = rowan.rotate(view_orientation, [first_k_value, 0, 0])
            right_center_k_vector = rowan.rotate(view_orientation, [last_k_value, 0, 0])

            npt.assert_allclose(dp.k_vectors[center, 0], top_center_k_vector)
            npt.assert_allclose(dp.k_vectors[center, -1], bottom_center_k_vector)
            npt.assert_allclose(dp.k_vectors[0, center], left_center_k_vector)
            npt.assert_allclose(dp.k_vectors[-1, center], right_center_k_vector)

    def test_cubic_system(self):
        length = 1
        box, positions = freud.data.UnitCell.sc().generate_system(
            num_replicas=16, scale=length, sigma_noise=0.1 * length
        )
        # Pick a non-integer value for zoom, to ensure that peaks besides k=0
        # are not perfectly aligned on pixels.
        dp = freud.diffraction.DiffractionPattern(grid_size=512)
        dp.compute((box, positions), zoom=4.123)

        # Locate brightest areas of diffraction pattern
        # (intensity > threshold), and check that the ideal
        # diffraction peak locations, given by k * R = 2*pi*N
        # for some lattice vector R and integer N, are contained
        # within these regions.
        # This test only checks N in range [-2, 2].
        threshold = 0.2
        xs, ys = np.nonzero(dp.diffraction > threshold)
        xy = np.dstack((xs, ys))[0]

        ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

        lattice_vector = np.array([length, length, length])
        for peak in ideal_peaks:
            for x, y in xy:
                k_vector = dp.k_vectors[x, y]
                dot_prod = np.dot(k_vector, lattice_vector)

                if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                    ideal_peaks[peak] = True

        assert all(ideal_peaks.values())

    def test_cubic_system_parameterized(self):
        length = 1
        box, positions = freud.data.UnitCell.sc().generate_system(
            num_replicas=16, scale=length, sigma_noise=0.1 * length
        )
        # Same as above test but with different grid_size,
        # output_size, and zoom values.
        for grid_size in (256, 1024):
            for output_size in (255, 256, 1023, 1024):
                for zoom in (1, 2.5, 4.123):
                    dp = freud.diffraction.DiffractionPattern(
                        grid_size=grid_size,
                        output_size=output_size,
                    )
                    dp.compute((box, positions), zoom=zoom)

                    threshold = 0.2
                    xs, ys = np.nonzero(dp.diffraction > threshold)
                    xy = np.dstack((xs, ys))[0]

                    ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

                    lattice_vector = np.array([length, length, length])
                    for peak in ideal_peaks:
                        for x, y in xy:
                            k_vector = dp.k_vectors[x, y]
                            dot_prod = np.dot(k_vector, lattice_vector)

                            if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                                ideal_peaks[peak] = True

                    assert all(ideal_peaks.values())

    def test_system_long_and_short_in_view_direction(self):
        # Same as above tests, but with unit cell lengthed
        # and shortened in the direction parallel to the view axis.
        # This should produce a diffraction pattern identical to the
        # simple cubic case.

        dp = freud.diffraction.DiffractionPattern(grid_size=512)

        for Lz in [0.5, 2, 3.456]:
            Lx, Ly = 1, 1
            cell = freud.box.Box(Lx=Lx, Ly=Ly, Lz=Lz)
            box, points = freud.data.UnitCell(cell).generate_system(
                num_replicas=16, scale=1, sigma_noise=0.1
            )

            dp.compute((box, points), zoom=2, view_orientation=None)

            threshold = 0.2
            xs, ys = np.nonzero(dp.diffraction > threshold)
            xy = np.dstack((xs, ys))[0]

            ideal_peaks = {i: False for i in [-2, -1, 0, 1, 2]}

            lattice_vector = np.array([Lx, Ly, Lz])
            for peak in ideal_peaks:
                for x, y in xy:
                    k_vector = dp.k_vectors[x, y]
                    dot_prod = np.dot(k_vector, lattice_vector)

                    if np.isclose(dot_prod, peak * 2 * np.pi, atol=1e-2):
                        ideal_peaks[peak] = True

            assert all(ideal_peaks.values())

    def test_unique_box_vector_lengths(self):
        # TODO: k-values, k-vectors and peaks should be
        # correct for various boxes with unique side lengths.
        # Testing w/ variations of other parameters is also possible.
        pass

    def test_rotated_system(self):
        # TODO: Same as above, but for boxes that are rotated
        # such that no face normal vector aligns with the view axis.
        pass

    def test_multiple_faces_project_equally(self):
        # TODO: Special case of above, test configurations where two
        # or three box face normal vectors project equally onto
        # the view axis.
        pass
