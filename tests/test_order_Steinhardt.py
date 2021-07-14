import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
import rowan
import util

import freud

matplotlib.use("agg")

# Validated against manual calculation and pyboo
PERFECT_FCC_Q6 = 0.57452416
PERFECT_FCC_W6 = -0.00262604


class TestSteinhardt:
    def test_shape(self):
        N = 1000
        L = 10

        box, positions = freud.data.make_random_system(L, N)

        comp = freud.order.Steinhardt(6)
        comp.compute((box, positions), neighbors={"r_max": 1.5})

        npt.assert_equal(comp.particle_order.shape[0], N)

    @pytest.mark.parametrize("sph_l", range(3, 8))
    def test_qlmi(self, sph_l):
        """Test the raw calculated qlmi."""
        special = pytest.importorskip("scipy.special")
        sph_harm = special.sph_harm

        atol = 1e-4
        L = 8
        N = 100
        box, points = freud.data.make_random_system(L, N)

        num_neighbors = 4

        # Note the order of m values provided by fsph.
        ms = np.array(list(range(sph_l + 1)) + [-m for m in range(1, sph_l + 1)])[
            :, np.newaxis
        ]

        aq = freud.locality.AABBQuery(box, points)
        nl = aq.query(
            points, {"exclude_ii": True, "num_neighbors": num_neighbors}
        ).toNeighborList()

        comp = freud.order.Steinhardt(sph_l)
        comp.compute(aq, neighbors=nl)
        qlmi = np.zeros([N, 2 * sph_l + 1], dtype=complex)

        # Loop over the particles and compute the qlmis for each (the
        # broadcasting syntax becomes rather abstruse for 3D arrays, and we
        # have to match the indices to the NeighborList anyway).
        for i in range(N):
            neighbors_i = nl[nl.query_point_indices == i]
            bonds = box.wrap(points[neighbors_i[:, 1]] - points[neighbors_i[:, 0]])
            r = np.linalg.norm(bonds, axis=-1)
            thetas = np.arccos(bonds[:, 2] / r)
            phis = np.arctan2(bonds[:, 1], bonds[:, 0])

            qlmi[i, :] = np.sum(
                sph_harm(ms, sph_l, phis[np.newaxis, :], thetas[np.newaxis, :]), axis=-1
            )

        qlmi /= num_neighbors
        npt.assert_allclose(comp.particle_harmonics, qlmi, atol=atol)

    def test_l_axis_aligned(self):
        # This test has three points along the z-axis. By construction, the
        # points on the end should have Q_l = 1 for odd l and the central
        # point should have Q_l = 0 for odd l. All three points should
        # have perfect order for even l.
        box = freud.box.Box.cube(10)
        positions = [[0, 0, -1], [0, 0, 0], [0, 0, 1]]

        for odd_l in range(1, 20, 2):
            comp = freud.order.Steinhardt(odd_l)
            comp.compute((box, positions), neighbors={"num_neighbors": 2})
            npt.assert_allclose(comp.particle_order, [1, 0, 1], atol=1e-5)

        for even_l in range(0, 20, 2):
            comp = freud.order.Steinhardt(even_l)
            comp.compute((box, positions), neighbors={"num_neighbors": 2})
            npt.assert_allclose(comp.particle_order, 1, atol=1e-5)

    def test_identical_environments_ql(self):
        box, positions = freud.data.UnitCell.fcc().generate_system(4, scale=2)
        r_max = 1.5
        test_set = util.make_raw_query_nlist_test_set(
            box, positions, positions, "ball", r_max, 0, True
        )
        for nq, neighbors in test_set:
            comp = freud.order.Steinhardt(6)
            comp.compute(nq, neighbors=neighbors)
            npt.assert_allclose(
                np.average(comp.particle_order), PERFECT_FCC_Q6, atol=1e-5
            )
            npt.assert_allclose(comp.particle_order, comp.particle_order[0], atol=1e-5)
            assert abs(comp.order - PERFECT_FCC_Q6) < 1e-5

            comp = freud.order.Steinhardt(6, average=True)
            comp.compute(nq, neighbors=neighbors)
            npt.assert_allclose(
                np.average(comp.particle_order), PERFECT_FCC_Q6, atol=1e-5
            )
            npt.assert_allclose(comp.particle_order, comp.particle_order[0], atol=1e-5)
            assert abs(comp.order - PERFECT_FCC_Q6) < 1e-5

    def test_identical_environments_ql_near(self):
        box, positions = freud.data.UnitCell.fcc().generate_system(4)

        r_max = 1.5
        n = 12
        test_set = util.make_raw_query_nlist_test_set(
            box, positions, positions, "nearest", r_max, n, True
        )
        for nq, neighbors in test_set:
            comp = freud.order.Steinhardt(6)
            comp.compute(nq, neighbors=neighbors)
            npt.assert_allclose(
                np.average(comp.particle_order), PERFECT_FCC_Q6, atol=1e-5
            )
            npt.assert_allclose(comp.particle_order, comp.particle_order[0], atol=1e-5)
            assert abs(comp.order - PERFECT_FCC_Q6) < 1e-5

            comp = freud.order.Steinhardt(6, average=True)
            comp.compute(nq, neighbors=neighbors)
            npt.assert_allclose(
                np.average(comp.particle_order), PERFECT_FCC_Q6, atol=1e-5
            )
            npt.assert_allclose(comp.particle_order, comp.particle_order[0], atol=1e-5)
            assert abs(comp.order - PERFECT_FCC_Q6) < 1e-5

        # Perturb one position
        perturbed_positions = positions.copy()
        perturbed_positions[-1] += [0.1, 0, 0]

        test_set = util.make_raw_query_nlist_test_set(
            box, perturbed_positions, perturbed_positions, "nearest", r_max, n, True
        )
        # Ensure exactly 13 values change for the perturbed system
        for nq, neighbors in test_set:
            comp = freud.order.Steinhardt(6)
            comp.compute(nq, neighbors=neighbors)
            assert sum(~np.isclose(comp.ql, PERFECT_FCC_Q6, rtol=1e-6)) == 13

            # More than 13 particles should change for
            # ql averaged over neighbors
            comp = freud.order.Steinhardt(6, average=True)
            comp.compute(nq, neighbors=neighbors)
            assert sum(~np.isclose(comp.particle_order, PERFECT_FCC_Q6, rtol=1e-6)) > 13

    def test_identical_environments_wl(self):
        box, positions = freud.data.UnitCell.fcc().generate_system(4, scale=2)

        r_max = 1.5
        test_set = util.make_raw_query_nlist_test_set(
            box, positions, positions, "ball", r_max, 0, True
        )
        for nq, neighbors in test_set:
            comp = freud.order.Steinhardt(6, wl=True)
            comp.compute(nq, neighbors=neighbors)
            npt.assert_allclose(
                np.average(comp.particle_order), PERFECT_FCC_W6, atol=1e-5
            )
            npt.assert_allclose(comp.particle_order, comp.particle_order[0], atol=1e-5)
            assert abs(comp.order - PERFECT_FCC_W6) < 1e-5

            comp = freud.order.Steinhardt(6, wl=True, average=True)
            comp.compute(nq, neighbors=neighbors)
            npt.assert_allclose(
                np.average(comp.particle_order), PERFECT_FCC_W6, atol=1e-5
            )
            npt.assert_allclose(comp.particle_order, comp.particle_order[0], atol=1e-5)
            assert abs(comp.order - PERFECT_FCC_W6) < 1e-5

    def test_identical_environments_wl_near(self):
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        r_max = 1.5
        n = 12
        test_set = util.make_raw_query_nlist_test_set(
            box, positions, positions, "nearest", r_max, n, True
        )
        for nq, neighbors in test_set:
            comp = freud.order.Steinhardt(6, wl=True)
            comp.compute(nq, neighbors=neighbors)
            npt.assert_allclose(
                np.average(comp.particle_order), PERFECT_FCC_W6, atol=1e-5
            )
            npt.assert_allclose(comp.particle_order, comp.particle_order[0], atol=1e-5)
            assert abs(comp.order - PERFECT_FCC_W6) < 1e-5

            comp = freud.order.Steinhardt(6, wl=True, average=True)
            comp.compute(nq, neighbors=neighbors)
            npt.assert_allclose(
                np.average(comp.particle_order), PERFECT_FCC_W6, atol=1e-5
            )
            npt.assert_allclose(comp.particle_order, comp.particle_order[0], atol=1e-5)
            assert abs(comp.order - PERFECT_FCC_W6) < 1e-5

    def test_weighted(self):
        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        r_max = 1.5
        n = 12
        test_set = util.make_raw_query_nlist_test_set(
            box, positions, positions, "nearest", r_max, n, True
        )

        # Skip test sets without an explicit neighbor list
        for nq, neighbors in filter(
            lambda ts: type(ts[1]) == freud.locality.NeighborList, test_set
        ):
            nlist = neighbors

            for wt in [0, 0.1, 0.9, 1.1, 10, 1e6]:
                # Change the weight of the first bond for each particle
                weights = nlist.weights.copy()
                weights[nlist.segments] = wt
                weighted_nlist = freud.locality.NeighborList.from_arrays(
                    len(positions),
                    len(positions),
                    nlist.query_point_indices,
                    nlist.point_indices,
                    nlist.vectors,
                    weights,
                )

                comp = freud.order.Steinhardt(6, weighted=True)
                comp.compute(nq, neighbors=weighted_nlist)

                # Unequal neighbor weighting in a perfect FCC structure
                # appears to increase the Q6 order parameter
                npt.assert_array_less(PERFECT_FCC_Q6, comp.particle_order)
                npt.assert_allclose(
                    comp.particle_order, comp.particle_order[0], atol=1e-5
                )
                npt.assert_array_less(PERFECT_FCC_Q6, comp.order)

                # Ensure that W6 values are altered by changing the weights
                comp = freud.order.Steinhardt(6, wl=True, weighted=True)
                comp.compute(nq, neighbors=weighted_nlist)
                with pytest.raises(AssertionError):
                    npt.assert_allclose(
                        np.average(comp.particle_order), PERFECT_FCC_W6, rtol=1e-5
                    )
                with pytest.raises(AssertionError):
                    npt.assert_allclose(comp.order, PERFECT_FCC_W6, rtol=1e-5)

    def test_attribute_access(self):
        comp = freud.order.Steinhardt(6)

        with pytest.raises(AttributeError):
            comp.order
        with pytest.raises(AttributeError):
            comp.particle_order

        box, positions = freud.data.UnitCell.fcc().generate_system(4)
        comp.compute((box, positions), neighbors={"r_max": 1.5})

        comp.order
        comp.particle_order

    def test_compute_twice_norm(self):
        """Test that computing norm twice works as expected."""
        L = 5
        num_points = 100
        box, points = freud.data.make_random_system(L, num_points, seed=0)

        st = freud.order.Steinhardt(6)
        first_result = st.compute((box, points), neighbors={"r_max": 1.5}).order
        second_result = st.compute((box, points), neighbors={"r_max": 1.5}).order

        npt.assert_array_almost_equal(first_result, second_result)

    def test_rotational_invariance(self):
        box = freud.box.Box.cube(10)
        positions = np.array(
            [
                [0, 0, 0],
                [-1, -1, 0],
                [-1, 1, 0],
                [1, -1, 0],
                [1, 1, 0],
                [-1, 0, -1],
                [-1, 0, 1],
                [1, 0, -1],
                [1, 0, 1],
                [0, -1, -1],
                [0, -1, 1],
                [0, 1, -1],
                [0, 1, 1],
            ]
        )
        query_point_indices = np.zeros(len(positions) - 1, dtype=int)
        point_indices = np.arange(1, len(positions))
        nlist = freud.locality.NeighborList.from_arrays(
            len(positions),
            len(positions),
            query_point_indices,
            point_indices,
            box.wrap(positions[point_indices] - positions[query_point_indices]),
        )

        q6 = freud.order.Steinhardt(6)
        w6 = freud.order.Steinhardt(6, wl=True)

        q6.compute((box, positions), neighbors=nlist)
        q6_unrotated_order = q6.particle_order[0]
        w6.compute((box, positions), neighbors=nlist)
        w6_unrotated_order = w6.particle_order[0]

        for i in range(10):
            np.random.seed(i)
            quat = rowan.random.rand()
            positions_rotated = rowan.rotate(quat, positions)

            # Ensure Q6 is rotationally invariant
            q6.compute((box, positions_rotated), neighbors=nlist)
            npt.assert_allclose(q6.particle_order[0], q6_unrotated_order, rtol=1e-5)
            npt.assert_allclose(q6.particle_order[0], PERFECT_FCC_Q6, rtol=1e-5)

            # Ensure W6 is rotationally invariant
            w6.compute((box, positions_rotated), neighbors=nlist)
            npt.assert_allclose(w6.particle_order[0], w6_unrotated_order, rtol=1e-5)
            npt.assert_allclose(w6.particle_order[0], PERFECT_FCC_W6, rtol=1e-5)

    def test_repr(self):
        comp = freud.order.Steinhardt(6)
        assert str(comp) == str(eval(repr(comp)))
        # Use non-default arguments for all parameters
        comp = freud.order.Steinhardt(6, average=True, wl=True, weighted=True)
        assert str(comp) == str(eval(repr(comp)))

    def test_repr_png(self):
        L = 5
        num_points = 100
        box, points = freud.data.make_random_system(L, num_points, seed=0)
        st = freud.order.Steinhardt(6)

        with pytest.raises(AttributeError):
            st.plot()
        assert st._repr_png_() is None

        st.compute(system=(box, points), neighbors={"r_max": 1.5})
        st._repr_png_()
        plt.close("all")

    def test_no_neighbors(self):
        """Ensure that particles without neighbors are assigned NaN."""
        box = freud.box.Box.cube(10)
        positions = [(0, 0, 0)]
        comp = freud.order.Steinhardt(6)
        comp.compute((box, positions), neighbors={"r_max": 1.25})

        assert np.all(np.isnan(comp.particle_order))
        npt.assert_allclose(np.nan_to_num(comp.particle_order), 0)

    def test_multiple_l(self):
        """Test the raw calculated qlmi."""
        special = pytest.importorskip("scipy.special")
        sph_harm = special.sph_harm

        atol = 1e-4
        L = 8
        N = 100
        box, points = freud.data.make_random_system(L, N)

        num_neighbors = 4

        sph_l = list(range(3, 8))

        # Note the order of m values provided by fsph.
        ms_per_l = [
            np.array(list(range(l + 1)) + [-m for m in range(1, l + 1)])[:, np.newaxis]
            for l in sph_l
        ]

        aq = freud.locality.AABBQuery(box, points)
        nl = aq.query(
            points, {"exclude_ii": True, "num_neighbors": num_neighbors}
        ).toNeighborList()

        comp = freud.order.Steinhardt(sph_l)
        comp.compute(aq, neighbors=nl)
        qlmis = [np.zeros([N, 2 * l + 1], dtype=complex) for l in sph_l]

        # Loop over the particles and compute the qlmis for each (the
        # broadcasting syntax becomes rather abstruse for 3D arrays, and we
        # have to match the indices to the NeighborList anyway).
        for l, ms, qlmi in zip(sph_l, ms_per_l, qlmis):
            for i in range(N):
                neighbors_i = nl[nl.query_point_indices == i]
                bonds = box.wrap(points[neighbors_i[:, 1]] - points[neighbors_i[:, 0]])
                r = np.linalg.norm(bonds, axis=-1)
                thetas = np.arccos(bonds[:, 2] / r)
                phis = np.arctan2(bonds[:, 1], bonds[:, 0])

                qlmi[i, :] = np.sum(
                    sph_harm(ms, l, phis[np.newaxis, :], thetas[np.newaxis, :]), axis=-1
                )

            qlmi /= num_neighbors
        assert all(
            np.allclose(comp.particle_harmonics[i], qlmis[i], atol=atol)
            for i in range(len(sph_l))
        )
