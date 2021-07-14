import numpy as np
import numpy.testing as npt
import pytest

import freud.locality


class TestNeighborList:
    def setup_method(self):
        # Define values
        self.L = 10
        self.N = 40

        self.query_args = dict(
            mode="nearest", num_neighbors=6, r_guess=3, exclude_ii=True
        )

        # Initialize Box and cell list
        box, points = freud.data.make_random_system(self.L, self.N)
        self.nq = freud.locality.AABBQuery(box, points)
        self.nlist = self.nq.query(points, self.query_args).toNeighborList()

    def test_writable(self):
        # query_point_indices shouldn't be writable in general or users may
        # break the ordered property of the neighbor list
        with pytest.raises(ValueError):
            self.nlist.query_point_indices[:] = 0

        # if query_point_indices isn't writable, point_indices shouldn't be
        with pytest.raises(ValueError):
            self.nlist.point_indices[:] = 0

        # the weights array may be useful to write to, though
        # TODO: weights aren't writable since changing to ManagedArray
        with pytest.raises(ValueError):
            self.nlist.weights[18] = 3

    def test_filter_r_max(self):
        points2 = self.nq.points[: self.N // 2]
        filter_max_distance = 2.5

        self.nlist = self.nq.query(points2, self.query_args).toNeighborList()
        old_size = len(self.nlist)

        # Make sure weights are 1
        npt.assert_equal(self.nlist.weights, 1)

        # Compute neighbor pairs that we expect to be kept after the filter
        kept_neighbors = self.nlist[self.nlist.distances < filter_max_distance]

        self.nlist.filter_r(filter_max_distance)
        new_size = len(self.nlist)

        assert old_size > 0
        assert new_size > 0
        assert new_size <= old_size

        # Make sure weights are still 1 after resize
        npt.assert_equal(self.nlist.weights, 1)

        # Make sure distances are filtered
        npt.assert_array_less(
            self.nlist.distances, np.full(new_size, filter_max_distance)
        )

        npt.assert_equal(kept_neighbors, self.nlist)

    def test_filter_r_max_min(self):
        points2 = self.nq.points[: self.N // 2]
        filter_min_distance = 1.5
        filter_max_distance = 2.5

        self.nlist = self.nq.query(points2, self.query_args).toNeighborList()
        old_size = len(self.nlist)

        # Make sure weights are 1
        npt.assert_equal(self.nlist.weights, 1)

        # Compute neighbor pairs that we expect to be kept after the filter
        kept_neighbors = self.nlist[
            np.logical_and(
                self.nlist.distances < filter_max_distance,
                self.nlist.distances >= filter_min_distance,
            )
        ]

        self.nlist.filter_r(filter_max_distance, filter_min_distance)
        new_size = len(self.nlist)

        assert old_size > 0
        assert new_size > 0
        assert new_size <= old_size

        # Make sure weights are still 1 after resize
        npt.assert_equal(self.nlist.weights, 1)

        # Make sure distances are filtered
        npt.assert_array_less(
            self.nlist.distances, np.full(new_size, filter_max_distance)
        )
        npt.assert_array_less(
            np.full(new_size, filter_min_distance), self.nlist.distances
        )

        npt.assert_equal(kept_neighbors, self.nlist)

    def test_filter(self):
        old_size = len(self.nlist)
        filt = (
            self.nlist.point_indices.astype(np.int32)
            - self.nlist.query_point_indices.astype(np.int32)
        ) % 2 == 0
        self.nlist.filter(filt)
        assert len(self.nlist) <= old_size

        # should be able to further filter
        self.nlist.filter_r(2.5)

    def test_find_first_index(self):
        nlist = self.nlist
        for (idx, i) in enumerate(nlist.query_point_indices):
            assert nlist.find_first_index(i) <= idx

    def test_segments(self):
        ones = np.ones(len(self.nlist), dtype=np.float32)
        assert np.allclose(np.add.reduceat(ones, self.nlist.segments), 6)
        assert np.allclose(self.nlist.neighbor_counts, 6)

    def test_from_arrays(self):
        query_point_indices = [0, 0, 1, 2, 3]
        point_indices = [1, 2, 3, 0, 0]
        vectors = np.ones((len(query_point_indices), 3))

        # implicit weights
        nlist = freud.locality.NeighborList.from_arrays(
            4, 4, query_point_indices, point_indices, vectors
        )
        assert np.allclose(nlist.weights, 1)

        # explicit weights
        weights = np.ones(len(query_point_indices)) * 4.0
        nlist = freud.locality.NeighborList.from_arrays(
            4, 4, query_point_indices, point_indices, vectors, weights
        )
        assert np.allclose(nlist.weights, 4)

        # copy of existing nlist by arrays
        weights = np.random.rand(len(query_point_indices))
        nlist = freud.locality.NeighborList.from_arrays(
            4, 4, query_point_indices, point_indices, vectors, weights
        )
        nlist2 = freud.locality.NeighborList.from_arrays(
            4,
            4,
            nlist.query_point_indices,
            nlist.point_indices,
            nlist.vectors,
            nlist.weights,
        )
        npt.assert_equal(nlist.query_point_indices, nlist2.query_point_indices)
        npt.assert_equal(nlist.point_indices, nlist2.point_indices)
        npt.assert_equal(nlist.distances, nlist2.distances)
        npt.assert_equal(nlist.weights, nlist2.weights)
        npt.assert_equal(nlist.vectors, nlist2.vectors)
        npt.assert_equal(nlist.neighbor_counts, nlist2.neighbor_counts)
        npt.assert_equal(nlist.segments, nlist2.segments)

        # too few reference particles
        with pytest.raises(ValueError):
            freud.locality.NeighborList.from_arrays(
                3, 4, query_point_indices, point_indices, vectors
            )

        # too few target particles
        with pytest.raises(ValueError):
            freud.locality.NeighborList.from_arrays(
                4, 3, query_point_indices, point_indices, vectors
            )

        # query particles not sorted
        with pytest.raises(ValueError):
            freud.locality.NeighborList.from_arrays(
                4, 4, point_indices, query_point_indices, vectors
            )

        # mismatched array sizes
        with pytest.raises(ValueError):
            # wrong number of query point indices
            freud.locality.NeighborList.from_arrays(
                4, 4, query_point_indices[:-1], point_indices, vectors
            )
        with pytest.raises(ValueError):
            # wrong number of point indices
            freud.locality.NeighborList.from_arrays(
                4, 4, query_point_indices, point_indices[:-1], vectors
            )
        with pytest.raises(ValueError):
            # wrong number of vectors
            freud.locality.NeighborList.from_arrays(
                4, 4, query_point_indices, point_indices, vectors[:-1]
            )
        with pytest.raises(ValueError):
            # wrong number of weights
            freud.locality.NeighborList.from_arrays(
                4, 4, query_point_indices, point_indices, vectors, weights[:-1]
            )

    def test_indexing_empty(self):
        # Ensure that empty NeighborLists have the right shape
        nlist = self.nq.query(np.empty((0, 3)), self.query_args).toNeighborList()
        assert nlist[:].shape == (0, 2)

    def test_indexing_arrays(self):
        # Make sure indexing the NeighborList is the same as indexing arrays
        for i, (idx_i, idx_j) in enumerate(self.nlist):
            assert idx_i == self.nlist.query_point_indices[i]
            assert idx_j == self.nlist.point_indices[i]

        for i, j in self.nlist:
            assert i != j

    def test_nl_size(self):
        assert len(self.nlist) == len(self.nlist.query_point_indices)
        assert len(self.nlist) == len(self.nlist.point_indices)

    def test_index_error(self):
        with pytest.raises(IndexError):
            nbonds = len(self.nlist)
            self.nlist[nbonds + 1]

    def test_index_writable(self):
        with pytest.raises(TypeError):
            self.nlist[:, 0] = 0

        with pytest.raises(TypeError):
            self.nlist[1, :] = 0

        with pytest.raises(TypeError):
            self.nlist[:] = 0

        with pytest.raises(TypeError):
            self.nlist[0, 0] = 0

    def test_copy(self):
        nlist = self.nlist
        nlist2 = nlist.copy()
        npt.assert_equal(nlist[:], nlist2[:])
        npt.assert_equal(nlist.distances, nlist2.distances)
        npt.assert_equal(nlist.weights, nlist2.weights)
        npt.assert_equal(nlist.segments, nlist2.segments)
        npt.assert_equal(nlist.neighbor_counts, nlist2.neighbor_counts)

    def test_ordering_default(self):
        # default behavior sorts by (i, j, distance)
        tuples = list(
            zip(
                self.nlist.query_point_indices,
                self.nlist.point_indices,
                self.nlist.distances,
            )
        )
        sorted_tuples = list(sorted(tuples))

        assert tuples == sorted_tuples

    def test_ordering_distance(self):
        nlist = self.nq.query(self.nq.points, self.query_args).toNeighborList(True)

        # test sorting by (i, distance, j)
        tuples = list(
            zip(nlist.query_point_indices, nlist.distances, nlist.point_indices)
        )
        sorted_tuples = list(sorted(tuples))

        assert tuples == sorted_tuples

    def test_num_points(self):
        query_point_indices = [0, 0, 1, 2, 3]
        point_indices = [1, 2, 3, 0, 0]
        vectors = np.ones((len(query_point_indices), 3))

        # test num_query_points and num_points when built from arrays
        nlist = freud.locality.NeighborList.from_arrays(
            42, 99, query_point_indices, point_indices, vectors
        )
        assert nlist.num_query_points == 42
        assert nlist.num_points == 99

        # test num_query_points and num_points when built from a query
        nlist = self.nq.query(self.nq.points[:-1], self.query_args).toNeighborList()
        assert nlist.num_query_points == len(self.nq.points) - 1
        assert nlist.num_points == len(self.nq.points)
