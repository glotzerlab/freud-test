import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
import util

import freud

matplotlib.use("agg")


class TestHexatic:
    def test_getK(self):
        hop = freud.order.Hexatic()
        npt.assert_equal(hop.k, 6)

    def test_getK_pass(self):
        k = 3
        hop = freud.order.Hexatic(k)
        npt.assert_equal(hop.k, 3)

    def test_order_size(self):
        boxlen = 10
        N = 500
        box, points = freud.data.make_random_system(boxlen, N, is2D=True)
        hop = freud.order.Hexatic()
        hop.compute((box, points))
        npt.assert_equal(len(hop.particle_order), N)

    def test_compute_random(self):
        boxlen = 10
        N = 500
        box, points = freud.data.make_random_system(boxlen, N, is2D=True)
        hop = freud.order.Hexatic()
        hop.compute((box, points))
        npt.assert_allclose(np.mean(hop.particle_order), 0.0 + 0.0j, atol=1e-1)

    def test_compute(self):
        boxlen = 10
        r_max = 3
        box = freud.box.Box.square(boxlen)
        points = [[0.0, 0.0, 0.0]]

        for i in range(6):
            points.append(
                [
                    np.cos(float(i) * 2.0 * np.pi / 6.0),
                    np.sin(float(i) * 2.0 * np.pi / 6.0),
                    0.0,
                ]
            )

        points = np.asarray(points, dtype=np.float32)
        points[:, 2] = 0.0
        hop = freud.order.Hexatic()

        # Test access
        hop.k
        with pytest.raises(AttributeError):
            hop.particle_order

        test_set = util.make_raw_query_nlist_test_set(
            box, points, points, "nearest", r_max, 6, True
        )
        for nq, neighbors in test_set:
            hop.compute(nq, neighbors=neighbors)
            # Test access
            hop.k
            hop.particle_order

            npt.assert_allclose(hop.particle_order[0], 1.0 + 0.0j, atol=1e-1)

    def test_weighted_random(self):
        boxlen = 10
        N = 5000
        box, points = freud.data.make_random_system(boxlen, N, is2D=True, seed=100)
        voro = freud.locality.Voronoi()
        voro.compute(system=(box, points))

        # Ensure that \psi'_0 is 1
        hop = freud.order.Hexatic(k=0, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(hop.particle_order, 1.0, atol=1e-6)

        # Ensure that \psi'_1 is 0
        hop = freud.order.Hexatic(k=1, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(np.absolute(hop.particle_order), 0.0, atol=1e-4)

        for k in range(0, 12):
            # Ensure that \psi'_k is between 0 and 1
            hop = freud.order.Hexatic(k=k, weighted=True)
            hop.compute(system=(box, points), neighbors=voro.nlist)
            order = np.absolute(hop.particle_order)
            assert (order >= 0).all() and (order <= 1).all()

            # Perform an explicit calculation in NumPy to verify results
            psi_k_weighted = np.zeros(len(points), dtype=np.complex128)
            total_weights = np.zeros(len(points))
            rijs = box.wrap(
                points[voro.nlist.point_indices]
                - points[voro.nlist.query_point_indices]
            )
            thetas = np.arctan2(rijs[:, 1], rijs[:, 0])
            total_weights, _ = np.histogram(
                voro.nlist.query_point_indices,
                bins=len(points),
                weights=voro.nlist.weights,
            )
            psi_k_weighted, _ = np.histogram(
                voro.nlist.query_point_indices,
                bins=len(points),
                weights=voro.nlist.weights * np.exp(thetas * k * 1.0j),
            )
            psi_k_weighted /= total_weights

            npt.assert_allclose(hop.particle_order, psi_k_weighted, atol=1e-5)

    def test_weighted_square(self):
        unitcell = freud.data.UnitCell.square()
        box, points = unitcell.generate_system(num_replicas=10)
        voro = freud.locality.Voronoi()
        voro.compute(system=(box, points))

        # Ensure that \psi'_4 is 1
        hop = freud.order.Hexatic(k=4, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(np.absolute(hop.particle_order), 1.0, atol=1e-5)

        # Ensure that \psi'_6 is 0
        hop = freud.order.Hexatic(k=6, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(np.absolute(hop.particle_order), 0.0, atol=1e-5)

    def test_weighted_hex(self):
        unitcell = freud.data.UnitCell.hex()
        box, points = unitcell.generate_system(num_replicas=10)
        voro = freud.locality.Voronoi()
        voro.compute(system=(box, points))

        # Ensure that \psi'_6 is 1
        hop = freud.order.Hexatic(k=6, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(np.absolute(hop.particle_order), 1.0, atol=1e-5)

        # Ensure that \psi'_4 is 0
        hop = freud.order.Hexatic(k=4, weighted=True)
        hop.compute(system=(box, points), neighbors=voro.nlist)
        npt.assert_allclose(np.absolute(hop.particle_order), 0.0, atol=1e-5)

    def test_normalization(self):
        """Verify normalizations for weighted and unweighted systems."""
        box = freud.Box.square(L=10)
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.31, 0.0],
                [0.17, 1.0, 0.0],
                [-1.0, 0.25, 0.0],
                [0.13, -1.0, 0.0],
            ]
        )
        query_point_indices = np.array([0, 0, 0, 0])
        point_indices = np.array([1, 2, 3, 4])
        rijs = box.wrap(points[point_indices] - points[query_point_indices])
        distances = np.linalg.norm(rijs, axis=-1)
        thetas = np.arctan2(rijs[:, 1], rijs[:, 0])
        weights = np.array([1, 0.7, 0.3, 0])
        nlist = freud.NeighborList.from_arrays(
            len(points),
            len(points),
            query_point_indices,
            point_indices,
            distances,
            weights,
        )
        for k in range(0, 12):
            # Unweighted calculation - normalized by number of neighbors
            psi_k = np.sum(np.exp(thetas * k * 1.0j)) / len(nlist)
            hop = freud.order.Hexatic(k=k)
            hop.compute(system=(box, points), neighbors=nlist)
            npt.assert_allclose(psi_k, hop.particle_order[0], atol=1e-5)
            # Weighted calculation - normalized by total neighbor weight
            psi_k_weighted = np.sum(nlist.weights * np.exp(thetas * k * 1.0j))
            psi_k_weighted /= np.sum(nlist.weights)
            hop_weighted = freud.order.Hexatic(k=k, weighted=True)
            hop_weighted.compute(system=(box, points), neighbors=nlist)
            npt.assert_allclose(
                psi_k_weighted, hop_weighted.particle_order[0], atol=1e-5
            )

    def test_3d_box(self):
        boxlen = 10
        N = 500
        box, points = freud.data.make_random_system(boxlen, N, is2D=False)
        hop = freud.order.Hexatic()
        with pytest.raises(ValueError):
            hop.compute((box, points))

    def test_repr(self):
        hop = freud.order.Hexatic(3)
        assert str(hop) == str(eval(repr(hop)))

        hop = freud.order.Hexatic(7, weighted=True)
        assert str(hop) == str(eval(repr(hop)))

    def test_repr_png(self):
        boxlen = 10
        N = 500
        box, points = freud.data.make_random_system(boxlen, N, is2D=True)
        hop = freud.order.Hexatic()
        with pytest.raises(AttributeError):
            hop.plot()
        assert hop._repr_png_() is None

        hop.compute((box, points))
        hop._repr_png_()
        hop.plot()
        plt.close("all")

    def test_no_neighbors(self):
        """Ensure that particles without neighbors are assigned NaN"""
        box = freud.box.Box.square(10)
        positions = [(0, 0, 0)]
        hop = freud.order.Hexatic()
        hop.compute((box, positions), neighbors={"r_max": 1.25})

        assert np.all(np.isnan(hop.particle_order))
        npt.assert_allclose(np.nan_to_num(hop.particle_order), 0)
