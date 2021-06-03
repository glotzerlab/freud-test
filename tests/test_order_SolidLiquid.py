import matplotlib
import numpy as np
import numpy.testing as npt
import pytest

import freud

matplotlib.use("agg")


class TestSolidLiquid:
    def test_shape(self):
        N = 1000
        L = 10

        box, positions = freud.data.make_random_system(L, N)

        comp = freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=6)
        comp.compute((box, positions), neighbors=dict(r_max=2.0))

        npt.assert_equal(comp.cluster_idx.shape, (N,))

    def test_nlist(self):
        """Check that the internally generated NeighborList is correct."""
        N = 1000
        L = 10

        box, positions = freud.data.make_random_system(L, N)

        query_args = dict(r_max=2.0, exclude_ii=True)
        comp = freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=6)
        comp.compute((box, positions), neighbors=query_args)

        aq = freud.locality.AABBQuery(box, positions)
        nlist = aq.query(positions, query_args).toNeighborList()

        npt.assert_array_equal(nlist[:], comp.nlist[:])

    def test_identical_environments(self):
        box, positions = freud.data.UnitCell.fcc().generate_system(4, scale=2)

        comp_default = freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=6)
        comp_no_norm = freud.order.SolidLiquid(
            6, q_threshold=0.3, solid_threshold=6, normalize_q=False
        )

        for comp in (comp_default, comp_no_norm):
            for query_args in (dict(r_max=2.0), dict(num_neighbors=12)):
                comp.compute((box, positions), neighbors=query_args)
                assert comp.largest_cluster_size == len(positions)
                assert len(comp.cluster_sizes) == 1
                assert comp.cluster_sizes[0] == len(positions)
                npt.assert_array_equal(comp.num_connections, 12)

    def test_attribute_access(self):
        box, positions = freud.data.UnitCell.fcc().generate_system(4, scale=2)
        sph_l = 6
        q_threshold = 0.7
        solid_threshold = 4
        normalize_q = False

        comp = freud.order.SolidLiquid(
            sph_l,
            q_threshold=q_threshold,
            solid_threshold=solid_threshold,
            normalize_q=normalize_q,
        )

        assert comp.l == sph_l
        npt.assert_allclose(comp.q_threshold, q_threshold)
        npt.assert_allclose(comp.solid_threshold, solid_threshold)
        assert comp.normalize_q == normalize_q

        with pytest.raises(AttributeError):
            comp.largest_cluster_size
        with pytest.raises(AttributeError):
            comp.cluster_sizes
        with pytest.raises(AttributeError):
            comp.cluster_idx
        with pytest.raises(AttributeError):
            comp.num_connections
        with pytest.raises(AttributeError):
            comp.ql_ij
        with pytest.raises(AttributeError):
            comp.plot()

        comp.compute((box, positions), neighbors=dict(r_max=2.0))

        comp.largest_cluster_size
        comp.cluster_sizes
        comp.cluster_idx
        comp.num_connections
        comp.ql_ij
        comp._repr_png_()

    def test_repr(self):
        comp = freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=6)
        assert str(comp) == str(eval(repr(comp)))

    def test_no_neighbors(self):
        """Ensure that particles without neighbors are assigned NaN"""
        box = freud.box.Box.cube(10)
        positions = [(0, 0, 0)]
        comp = freud.order.SolidLiquid(6, q_threshold=0.7, solid_threshold=6)
        comp.compute((box, positions), neighbors=dict(r_max=2.0))

        npt.assert_equal(comp.cluster_idx, np.arange(len(comp.cluster_idx)))
        npt.assert_equal(comp.cluster_sizes, np.ones_like(comp.cluster_sizes))
        assert comp.largest_cluster_size == 1
        npt.assert_equal(comp.num_connections, np.zeros_like(comp.cluster_sizes))
        assert np.all(np.isnan(comp.particle_harmonics))
        npt.assert_equal(comp.ql_ij, [])
