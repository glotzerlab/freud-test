import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

import freud

matplotlib.use("agg")


class TestCluster:
    # by Chrisy
    test_folder = os.path.join(os.path.dirname(__file__), "numpy_test_files")

    def test_single_cluster(self):
        fn = os.path.join(self.test_folder, "bcc.npy")
        xyz = np.load(fn)
        xyz = np.array(xyz, dtype=np.float32)
        xyz.flags["WRITEABLE"] = False
        L = np.max(xyz) * 2
        box = freud.box.Box.cube(L)

        r_max = 3.1
        num_neighbors = 14
        threshold_prefactor = 0.1
        threshold = threshold_prefactor * r_max

        match = freud.environment.EnvironmentCluster()
        with pytest.raises(AttributeError):
            match.point_environments
        with pytest.raises(AttributeError):
            match.num_particles
        with pytest.raises(AttributeError):
            match.num_clusters
        with pytest.raises(AttributeError):
            match.cluster_idx
        with pytest.raises(AttributeError):
            match.cluster_environments

        query_args = dict(r_guess=r_max, num_neighbors=num_neighbors)
        match.compute((box, xyz), threshold, neighbors=query_args)

        cluster_env = match.cluster_environments

        fn = os.path.join(self.test_folder, "bcc_env.npy")
        bcc_env = np.load(fn)
        num_cluster = len(cluster_env)
        env_cluster = cluster_env[0]

        # This is a nifty way of lexicographically sorting our arrays, for the
        # purposes of an element-by-element comparison.
        # np.lexsort() sorts by the columns you feed it, with the final fed
        # column being the "primary" sorting key.
        # cluster_environments might provide the motif with its vectors sorted
        # any old way, and if we compare against a saved numpy array then we
        # have to order the two arrays in the same fashion.
        sorted_env_cluster = env_cluster[
            np.lexsort((env_cluster[:, 0], env_cluster[:, 1], env_cluster[:, 2]))
        ]
        sorted_bcc_env = bcc_env[
            np.lexsort((bcc_env[:, 0], bcc_env[:, 1], bcc_env[:, 2]))
        ]
        npt.assert_equal(num_cluster, 1, err_msg="Number of BCC cluster fail")
        npt.assert_allclose(
            sorted_env_cluster,
            sorted_bcc_env,
            atol=1e-5,
            err_msg="BCC Cluster Environment fail",
        )

    # by Chrisy
    def test_multi_cluster(self):
        fn = os.path.join(self.test_folder, "sc.npy")
        xyz = np.load(fn)
        xyz = np.array(xyz, dtype=np.float32)
        box = freud.box.Box.cube(21)

        r_max = 4
        num_neighbors = 6
        threshold_prefactor = 0.1
        threshold = threshold_prefactor * r_max

        match = freud.environment.EnvironmentCluster()
        query_args = dict(r_guess=r_max, num_neighbors=num_neighbors)
        match.compute((box, xyz), threshold, neighbors=query_args)

        cluster_env = match.cluster_environments

        fn = os.path.join(self.test_folder, "sc_env.npy")
        sc_env = np.load(fn)
        # Randomly choose the 3rd cluster here to test
        env_cluster = cluster_env[2]
        num_cluster = len(cluster_env)

        # This is a nifty way of lexicographically sorting our arrays, for the
        # purposes of an element-by-element comparison.
        # np.lexsort() sorts by the columns you feed it, with the final fed
        # column being the "primary" sorting key.
        # cluster_environments might provide the motif with its vectors sorted
        # any old way, and if we compare against a saved numpy array then we
        # have to order the two arrays in the same fashion.
        sorted_env_cluster = env_cluster[
            np.lexsort((env_cluster[:, 0], env_cluster[:, 1], env_cluster[:, 2]))
        ]
        sorted_sc_env = sc_env[np.lexsort((sc_env[:, 0], sc_env[:, 1], sc_env[:, 2]))]
        npt.assert_equal(num_cluster, 6, err_msg="Number of SC cluster fail")
        npt.assert_allclose(
            sorted_env_cluster,
            sorted_sc_env,
            atol=1e-5,
            err_msg="SC Cluster Environment fail",
        )

    # Test EnvironmentCluster.compute function, defining clusters using
    # constant k neighbors, hard_r=false, registration=false
    def test_cluster_kNeighbor(self):
        fn = os.path.join(self.test_folder, "bcc.npy")
        xyz = np.load(fn)
        xyz = np.array(xyz, dtype=np.float32)
        L = np.max(xyz) * 2
        box = freud.box.Box.cube(L)

        r_max = 3.1
        num_neighbors = 14
        threshold_prefactor = 0.1
        threshold = threshold_prefactor * r_max

        match = freud.environment.EnvironmentCluster()
        query_args = dict(r_guess=r_max, num_neighbors=num_neighbors)
        match.compute((box, xyz), threshold, registration=False, neighbors=query_args)

        cluster_env = match.cluster_environments

        fn = os.path.join(self.test_folder, "bcc_env.npy")
        bcc_env = np.load(fn)
        num_cluster = len(cluster_env)
        env_cluster = cluster_env[0]

        # This is a nifty way of lexicographically sorting our arrays, for the
        # purposes of an element-by-element comparison.
        # np.lexsort() sorts by the columns you feed it, with the final fed
        # column being the "primary" sorting key.
        # cluster_environments might provide the motif with its vectors sorted
        # any old way, and if we compare against a saved numpy array then we
        # have to order the two arrays in the same fashion.
        sorted_env_cluster = env_cluster[
            np.lexsort((env_cluster[:, 0], env_cluster[:, 1], env_cluster[:, 2]))
        ]
        sorted_bcc_env = bcc_env[
            np.lexsort((bcc_env[:, 0], bcc_env[:, 1], bcc_env[:, 2]))
        ]
        npt.assert_equal(num_cluster, 1, err_msg="Number of BCC cluster fail")
        npt.assert_allclose(
            sorted_env_cluster,
            sorted_bcc_env,
            atol=1e-5,
            err_msg="BCC Cluster Environment fail",
        )

    # Test EnvironmentCluster.compute function, hard_r=true, registration=false
    def test_cluster_hardr(self):
        fn = os.path.join(self.test_folder, "bcc.npy")
        xyz = np.load(fn)
        xyz = np.array(xyz, dtype=np.float32)
        L = np.max(xyz) * 2
        box = freud.box.Box.cube(L)

        r_max = 3.1
        num_neighbors = 14
        threshold_prefactor = 0.1
        threshold = threshold_prefactor * r_max

        match = freud.environment.EnvironmentCluster()
        query_args = dict(r_max=r_max, num_neighbors=num_neighbors)
        match.compute((box, xyz), threshold, registration=False, neighbors=query_args)
        cluster_env = match.cluster_environments

        fn = os.path.join(self.test_folder, "bcc_env.npy")
        bcc_env = np.load(fn)
        num_cluster = len(cluster_env)
        env_cluster = cluster_env[0]

        # This is a nifty way of lexicographically sorting our arrays, for the
        # purposes of an element-by-element comparison.
        # np.lexsort() sorts by the columns you feed it, with the final fed
        # column being the "primary" sorting key.
        # cluster_environments might provide the motif with its vectors sorted
        # any old way, and if we compare against a saved numpy array then we
        # have to order the two arrays in the same fashion.
        sorted_env_cluster = env_cluster[
            np.lexsort((env_cluster[:, 0], env_cluster[:, 1], env_cluster[:, 2]))
        ]
        sorted_bcc_env = bcc_env[
            np.lexsort((bcc_env[:, 0], bcc_env[:, 1], bcc_env[:, 2]))
        ]
        npt.assert_equal(num_cluster, 1, err_msg="Number of BCC cluster fail")
        npt.assert_allclose(
            sorted_env_cluster,
            sorted_bcc_env,
            atol=1e-5,
            err_msg="BCC Cluster Environment fail",
        )

    # Test EnvironmentCluster.compute function,
    # hard_r=false, registration=true, global=true
    def test_cluster_registration(self):
        fn = os.path.join(self.test_folder, "sc_N54.npy")
        xyz = np.load(fn)
        xyz = np.array(xyz, dtype=np.float32)

        r_max = 4
        num_neighbors = 6
        threshold_prefactor = 0.005
        threshold = threshold_prefactor * r_max

        # Define rotation matrix, rotate along z axis by pi/24 degree
        rotationAngle = np.pi / 24.0
        R = np.array(
            [
                [np.cos(rotationAngle), -np.sin(rotationAngle), 0],
                [np.sin(rotationAngle), np.cos(rotationAngle), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        # Rotate particles that y>0, introduce grain boundary
        for i in range(len(xyz)):
            if xyz[i, 1] < 0.0:
                xyz[i] = R.dot(xyz[i])

        L = np.max(xyz) * 3.0
        box = freud.box.Box(L, L, L, 0, 0, 0)

        match = freud.environment.EnvironmentCluster()
        query_args = dict(r_guess=r_max, num_neighbors=num_neighbors)
        match.compute(
            (box, xyz),
            threshold,
            registration=True,
            global_search=True,
            neighbors=query_args,
        )
        clusters = match.cluster_idx

        # Get environment for each particle
        tot_env = match.point_environments

        # Particles with index 22 and 31 have opposite y positions,
        # they should have the same local environment
        npt.assert_equal(
            clusters[22],
            clusters[31],
            err_msg="two points do not have similar environment",
        )

        # Particle 22 and particle 31's local environments should match
        returnResult = freud.environment._is_similar_motif(
            box, tot_env[22], tot_env[31], 0.005, registration=True
        )
        npt.assert_equal(
            len(returnResult[1]),
            num_neighbors,
            err_msg="two environments are not similar",
        )

    # Test EnvironmentCluster._minimize_RMSD and registration functionality.
    # Overkill? Maybe.
    def test_minimize_RMSD(self):
        env_vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]])
        threshold_sq = 0.1

        # https://en.wikipedia.org/wiki/Rotation_matrix
        norm = np.array([1, 1, 1])
        norm = norm / np.sqrt(np.dot(norm, norm))

        theta = np.pi / 10

        ux = norm[0]
        uy = norm[1]
        uz = norm[2]
        c = np.cos(theta)
        s = np.sin(theta)

        R = np.array(
            [
                [
                    c + pow(ux, 2.0) * (1 - c),
                    ux * uy * (1 - c) - uz * s,
                    ux * uz * (1 - c) + uy * s,
                ],
                [
                    uy * ux * (1 - c) + uz * s,
                    c + pow(uy, 2.0) * (1 - c),
                    uy * uz * (1 - c) - ux * s,
                ],
                [
                    uz * ux * (1 - c) - uy * s,
                    uz * uy * (1 - c) + ux * s,
                    c + pow(uz, 2.0) * (1 - c),
                ],
            ]
        )

        # 1. Grab the environment vectors and set up a large enough box.
        e0 = np.array(env_vec, dtype=np.single)
        scale = 1.5
        rsq_arr = [np.dot(vec, vec) for vec in e0]
        rsq_max = max(rsq_arr)
        L = 2.0 * np.sqrt(rsq_max) * scale
        box = freud.box.Box.cube(L)
        # 2. Re-index the environment randomly to create a second environment.
        e1 = np.copy(e0)
        np.random.seed(0)
        np.random.shuffle(e1)
        # 3. Verify that OUR method _is_similar_motif gives that these two
        #    environments are similar.
        [refPoints2, isSim_vec_map] = freud.environment._is_similar_motif(
            box, e0, e1, threshold_sq, registration=False
        )
        npt.assert_allclose(
            e0, refPoints2[np.asarray(list(isSim_vec_map.values()))], atol=1e-6
        )
        # 4. Calculate the minimal RMSD.
        [min_rmsd, refPoints2, minRMSD_vec_map] = freud.environment._minimize_RMSD(
            box, e0, e1, registration=False
        )
        # 5. Verify that the _minimize_RMSD method finds 0 minimal RMSD
        #    (with no registration.)
        npt.assert_equal(0.0, min_rmsd)
        # 6. Verify that it gives the same vector mapping that
        # _is_similar_motif gave.
        npt.assert_equal(
            np.asarray(list(isSim_vec_map.values())),
            np.asarray(list(minRMSD_vec_map.values())),
        )
        npt.assert_allclose(
            e0, refPoints2[np.asarray(list(minRMSD_vec_map.values()))], atol=1e-6
        )
        # 7. Rotate the motif by a known rotation matrix. this matrix MUST be
        #    s.t. the minimal rmsd is the rmsd of the 1-1 mapping between the
        #    vectors of the pre-rotated environment and the post-rotated
        #    environment (with no index-mixing). This isn't guaranteed for any
        #    matrix, whatsoever. Work only with environments and rotations for
        #    which you know exactly what's going on here. Be careful here:
        #    R rotates the position matrix where positions are COLUMN vectors.
        e0_rot = np.array(np.transpose(np.dot(R, np.transpose(e0))), dtype=np.single)
        # 8. Calculate the minimal RMSD assuming the 1-1 relationship.
        delta = e0_rot - e0
        deltasq = np.array([np.dot(vec, vec) for vec in delta])
        deltasum = np.sum(deltasq)
        deltasum /= len(deltasq)
        analy_rmsd = np.sqrt(deltasum)
        # 9. Verify that _minimize_RMSD gives this minimal RMSD
        #    (with no registration).
        [min_rmsd, refPoints2, minRMSD_vec_map] = freud.environment._minimize_RMSD(
            box, e0, e0_rot, registration=False
        )
        npt.assert_allclose(analy_rmsd, min_rmsd, atol=1e-5)
        npt.assert_allclose(
            e0_rot, refPoints2[np.asarray(list(minRMSD_vec_map.values()))], atol=1e-5
        )
        # 10. Re-index the second environment randomly again.
        e1_rot = np.copy(e0_rot)
        np.random.shuffle(e1_rot)
        # 11. Verify that _minimize_RMSD gives this minimal RMSD again
        #     (with no registration).
        [min_rmsd, refPoints2, minRMSD_vec_map] = freud.environment._minimize_RMSD(
            box, e0, e1_rot, registration=False
        )
        npt.assert_allclose(analy_rmsd, min_rmsd, atol=1e-5)
        npt.assert_allclose(
            e0_rot, refPoints2[np.asarray(list(minRMSD_vec_map.values()))], atol=1e-5
        )
        # 12. Now use _minimize_RMSD with registration turned ON.
        [min_rmsd, refPoints2, minRMSD_vec_map] = freud.environment._minimize_RMSD(
            box, e0, e1_rot, registration=True
        )
        # 13. This should get us back to 0 minimal rmsd.
        npt.assert_allclose(0.0, min_rmsd, atol=1e-5)
        npt.assert_allclose(
            e0, refPoints2[np.asarray(list(minRMSD_vec_map.values()))], atol=1e-5
        )
        # 14. Finally use _is_similar_motif with registration turned ON.
        [refPoints2, isSim_vec_map] = freud.environment._is_similar_motif(
            box, e0, e1_rot, threshold_sq, registration=True
        )
        npt.assert_allclose(
            e0, refPoints2[np.asarray(list(isSim_vec_map.values()))], atol=1e-5
        )

    def test_repr(self):
        match = freud.environment.EnvironmentCluster()
        assert str(match) == str(eval(repr(match)))

    def test_repr_png(self):
        fn = os.path.join(self.test_folder, "bcc.npy")
        xyz = np.load(fn)
        xyz = np.array(xyz, dtype=np.float32)
        xyz.flags["WRITEABLE"] = False
        L = np.max(xyz) * 2

        r_max = 3.1
        num_neighbors = 14
        threshold_prefactor = 0.1
        threshold = threshold_prefactor * r_max

        box = freud.box.Box.square(L)
        xyz = np.load(fn)
        xyz = np.array(xyz, dtype=np.float32)
        xyz[:, 2] = 0
        xyz.flags["WRITEABLE"] = False
        match = freud.environment.EnvironmentCluster()

        with pytest.raises(AttributeError):
            match.plot()
        assert match._repr_png_() is None

        query_args = dict(r_guess=r_max, num_neighbors=num_neighbors)
        match.compute((box, xyz), threshold, neighbors=query_args)
        match._repr_png_()
        plt.close("all")


class TestEnvironmentMotifMatch:
    def test_square(self):
        """Test that a simple square motif correctly matches."""
        motif = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
        points = motif + [[0, 0, 0]]

        r_max = 1.5
        num_neighbors = 4

        box = freud.box.Box.square(3)
        match = freud.environment.EnvironmentMotifMatch()
        query_args = dict(r_guess=r_max, num_neighbors=num_neighbors)
        match.compute((box, points), motif, 0.1, neighbors=query_args)
        matches = match.matches

        for i in range(len(motif)):
            assert not matches[i]
        assert matches[len(motif)]


class TestEnvironmentRMSDMinimizer:
    def test_api(self):
        """This test simply verifies functional code, but not correctness."""
        motif = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
        points = motif + [[0, 0, 0]]

        r_max = 1.5
        num_neighbors = 4

        box = freud.box.Box.square(3)
        match = freud.environment._EnvironmentRMSDMinimizer()
        query_args = dict(r_guess=r_max, num_neighbors=num_neighbors)
        match.compute((box, points), motif, neighbors=query_args)
        assert np.all(match.rmsds[:-1] > 0)
        assert np.isclose(match.rmsds[-1], 0, atol=1e-6)
