import numpy as np
import numpy.testing as npt
import pytest
import rowan

import freud


class TestLocalBondProjection:
    def test_nlist(self):
        """Check that the internally generated NeighborList is correct."""
        boxlen = 10
        N = 500
        num_neighbors = 8
        r_guess = 3
        query_args = dict(num_neighbors=num_neighbors, r_guess=r_guess)

        N_query = N // 3

        box, points = freud.data.make_random_system(boxlen, N, is2D=True)
        _, query_points = freud.data.make_random_system(boxlen, N_query, is2D=True)
        ors = rowan.random.rand(N)
        proj_vecs = np.asarray([[0, 0, 1]])

        ang = freud.environment.LocalBondProjection()
        ang.compute((box, points), ors, proj_vecs, query_points, neighbors=query_args)

        aq = freud.locality.AABBQuery(box, points)
        nlist = aq.query(query_points, query_args).toNeighborList()

        npt.assert_array_equal(nlist[:], ang.nlist[:])

    def test_attribute_access(self):
        boxlen = 10
        N = 100
        num_neighbors = 8
        r_guess = 3
        query_args = dict(num_neighbors=num_neighbors, r_guess=r_guess)

        box, points = freud.data.make_random_system(boxlen, N, is2D=True)
        ors = rowan.random.rand(N)
        proj_vecs = np.asarray([[0, 0, 1]])

        ang = freud.environment.LocalBondProjection()

        with pytest.raises(AttributeError):
            ang.nlist
        with pytest.raises(AttributeError):
            ang.projections
        with pytest.raises(AttributeError):
            ang.normed_projections

        ang.compute((box, points), ors, proj_vecs, neighbors=query_args)

        ang.nlist
        ang.projections
        ang.normed_projections

    def test_compute(self):
        boxlen = 4
        num_neighbors = 1
        r_guess = 2
        query_args = dict(num_neighbors=num_neighbors, r_guess=r_guess)

        box = freud.box.Box.cube(boxlen)

        proj_vecs = np.asarray([[0, 0, 1]])

        # Create three points in an L-shape.
        points = [[0, 0, 0]]
        points.append([1, 0, 0])
        points.append([0, 0, 1.5])
        # Three orientations:
        # 1. The identity
        ors = [[1, 0, 0, 0]]
        # 2. A rotation about the y axis by pi/2
        ors.append([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
        # 3. A rotation about the z axis by pi/2
        ors.append([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])

        ors = np.asarray(ors, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)

        # First have no particle symmetry

        ang = freud.environment.LocalBondProjection()
        ang.compute((box, points), ors, proj_vecs, neighbors=query_args)

        nlist = freud.locality.AABBQuery(box, points).query(
            points, dict(num_neighbors=num_neighbors, r_guess=r_guess, exclude_ii=True)
        )
        bonds = [(i[0], i[1]) for i in nlist]

        # We will look at the bond between [1, 0, 0] as query_point
        # and [0, 0, 0] as point
        # This will give bond [-1, 0, 0].
        # Since [1, 0, 0] is the query_point at index 1, we rotate
        # this about y axis by pi/2, which will give
        # [0, 0, 1].
        # The projection onto [0, 0, 1] is cos(0) = 1.
        index = bonds.index((0, 1))
        npt.assert_allclose(ang.projections[index], 1, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[index], 1, atol=1e-6)

        # We will look at the bond between [0, 0, 0] as query_point
        # and [1, 0, 0] as point
        # This will give bond [-1, 0, 0].
        # Since [0, 0, 0] is the query_point at index 0, we rotate
        # this by the identity, which will give [-1, 0, 0].
        # The projection onto [0, 0, 1] is 0.
        index = bonds.index((1, 0))
        npt.assert_allclose(ang.projections[index], 0, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[index], 0, atol=1e-6)

        # We will look at the bond between [0, 0, 1.5] as query_point
        # and [0, 0, 0] as point
        # This will give bond [0, 0, -1.5].
        # Since [0, 0, 0] is the query_point at index 0, we rotate
        # this by the identity, which will give [0, 0, -1.5].
        # The projection onto [0, 0, 1] is -1.5.
        index = bonds.index((2, 0))
        npt.assert_allclose(ang.projections[index], -1.5, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[index], -1, atol=1e-6)

        # Specify that rotations about y by +/-pi/2 and rotations about x by pi
        # result in equivalent particle shapes
        qs = [
            [1, 0, 0, 0],
            [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
            [np.cos(np.pi / 2), np.sin(np.pi / 2), 0, 0],
        ]

        equiv_quats = []
        for q in qs:
            equiv_quats.append(q)
            # we have to include the adjoint (inverse) because this is a group
            equiv_quats.append(np.array([q[0], -q[1], -q[2], -q[3]]))
        equiv_quats = np.asarray(equiv_quats, dtype=np.float32)

        ang.compute((box, points), ors, proj_vecs, None, equiv_quats, query_args)

        # Now all projections should be cos(0)=1
        npt.assert_allclose(ang.projections[1], 1, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[1], 1, atol=1e-6)
        npt.assert_allclose(ang.projections[0], 1, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[0], 1, atol=1e-6)
        npt.assert_allclose(ang.projections[2], 1.5, atol=1e-6)
        npt.assert_allclose(ang.normed_projections[2], 1, atol=1e-6)

    def test_repr(self):
        ang = freud.environment.LocalBondProjection()
        assert str(ang) == str(eval(repr(ang)))
