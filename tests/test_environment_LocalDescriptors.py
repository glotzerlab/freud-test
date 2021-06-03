from functools import lru_cache

import numpy as np
import numpy.testing as npt
import pytest
from sympy.physics.wigner import wigner_3j

import freud


def get_ql(p, descriptors, nlist, weighted=False):
    """Given a set of points and a LocalDescriptors object (and the
    underlying neighborlist), compute the per-particle Steinhardt ql
    order parameter for all :math:`l` values up to the maximum quantum
    number used in the computation of the descriptors."""
    qbar_lm = np.zeros((p.shape[0], descriptors.sph.shape[1]), dtype=np.complex128)
    for i in range(p.shape[0]):
        indices = nlist.query_point_indices == i
        Ylms = descriptors.sph[indices, :]
        if weighted:
            weights = nlist.weights[indices, np.newaxis]
            weights /= np.sum(weights)
            num_neighbors = 1
        else:
            weights = np.ones_like(Ylms)
            num_neighbors = descriptors.sph.shape[0] / p.shape[0]
        qbar_lm[i, :] = np.sum(Ylms * weights, axis=0) / num_neighbors

    ql = np.zeros((qbar_lm.shape[0], descriptors.l_max + 1))
    for i in range(ql.shape[0]):
        for l in range(ql.shape[1]):
            for k in range(l ** 2, (l + 1) ** 2):
                ql[i, l] += np.absolute(qbar_lm[i, k]) ** 2
            ql[i, l] = np.sqrt(4 * np.pi / (2 * l + 1) * ql[i, l])

    return ql


def lm_index(l, m):
    return l ** 2 + (m if m >= 0 else l - m)


@lru_cache(maxsize=None)
def get_wigner3j(l, m1, m2, m3):
    return float(wigner_3j(l, l, l, m1, m2, m3))


def get_wl(p, descriptors, nlist):
    """Given a set of points and a LocalDescriptors object (and the
    underlying neighborlist), compute the per-particle Steinhardt wl
    order parameter for all :math:`l` values up to the maximum quantum
    number used in the computation of the descriptors."""
    qbar_lm = np.zeros((p.shape[0], descriptors.sph.shape[1]), dtype=np.complex128)

    num_neighbors = descriptors.sph.shape[0] / p.shape[0]
    for i in range(p.shape[0]):
        indices = nlist.query_point_indices == i
        qbar_lm[i, :] = np.sum(descriptors.sph[indices, :], axis=0) / num_neighbors

    wl = np.zeros((qbar_lm.shape[0], descriptors.l_max + 1), dtype=np.complex128)
    for i in range(wl.shape[0]):
        for l in range(wl.shape[1]):
            for m1 in range(-l, l + 1):
                for m2 in range(max(-l - m1, -l), min(l - m1, l) + 1):
                    m3 = -m1 - m2
                    # Manually add Condon-Shortley phase
                    phase = 1
                    for m in m1, m2, m3:
                        if m > 0 and m % 2 == 1:
                            phase *= -1
                    wl[i, l] += (
                        phase
                        * get_wigner3j(l, m1, m2, m3)
                        * qbar_lm[i, lm_index(l, m1)]
                        * qbar_lm[i, lm_index(l, m2)]
                        * qbar_lm[i, lm_index(l, m3)]
                    )
    return wl


class TestLocalDescriptors:
    def test_shape(self):
        N = 1000
        num_neighbors = 4
        l_max = 8
        L = 10

        box, positions = freud.data.make_random_system(L, N)
        positions.flags["WRITEABLE"] = False

        comp = freud.environment.LocalDescriptors(l_max, True)

        # Test access
        with pytest.raises(AttributeError):
            comp.sph
        with pytest.raises(AttributeError):
            comp.num_sphs

        comp.compute((box, positions), neighbors={"num_neighbors": num_neighbors})

        # Test access
        comp.sph
        comp.num_sphs

        assert comp.sph.shape[0] == N * num_neighbors

        assert comp.l_max == l_max

    def test_global(self):
        N = 1000
        num_neighbors = 4
        l_max = 8
        L = 10

        box, positions = freud.data.make_random_system(L, N)

        comp = freud.environment.LocalDescriptors(l_max, True, "global")
        comp.compute((box, positions), neighbors=dict(num_neighbors=num_neighbors))

        sphs = comp.sph

        assert sphs.shape[0] == N * num_neighbors

    def test_particle_local(self):
        N = 1000
        num_neighbors = 4
        l_max = 8
        L = 10

        box, positions = freud.data.make_random_system(L, N)
        orientations = np.random.uniform(-1, 1, size=(N, 4)).astype(np.float32)
        orientations /= np.sqrt(np.sum(orientations ** 2, axis=-1))[:, np.newaxis]

        comp = freud.environment.LocalDescriptors(l_max, True, mode="particle_local")

        with pytest.raises(RuntimeError):
            comp.compute((box, positions), neighbors=dict(num_neighbors=num_neighbors))

        comp.compute(
            (box, positions),
            orientations=orientations,
            neighbors=dict(num_neighbors=num_neighbors),
        )

        sphs = comp.sph

        assert sphs.shape[0] == N * num_neighbors

    def test_unknown_modes(self):
        N = 1000
        l_max = 8
        L = 10

        box, positions = freud.data.make_random_system(L, N)

        with pytest.raises(ValueError):
            freud.environment.LocalDescriptors(l_max, True, mode="particle_local_wrong")

    def test_nlist(self):
        """Check that the internally generated NeighborList is correct."""
        N = 1000
        num_neighbors = 4
        l_max = 8
        L = 10

        box, positions = freud.data.make_random_system(L, N)
        positions2 = np.random.uniform(-L / 2, L / 2, size=(N // 3, 3)).astype(
            np.float32
        )

        comp = freud.environment.LocalDescriptors(l_max, True)
        qargs = {"num_neighbors": num_neighbors}
        comp.compute((box, positions), positions2, neighbors=qargs)

        aq = freud.locality.AABBQuery(box, positions)
        nlist = aq.query(positions2, qargs).toNeighborList()

        npt.assert_array_equal(nlist[:], comp.nlist[:])

    def test_shape_twosets(self):
        N = 1000
        num_neighbors = 4
        l_max = 8
        L = 10

        box, positions = freud.data.make_random_system(L, N)
        positions2 = np.random.uniform(-L / 2, L / 2, size=(N // 3, 3)).astype(
            np.float32
        )

        comp = freud.environment.LocalDescriptors(l_max, True)
        comp.compute(
            (box, positions), positions2, neighbors={"num_neighbors": num_neighbors}
        )
        sphs = comp.sph
        assert sphs.shape[0] == N // 3 * num_neighbors

    def test_repr(self):
        comp = freud.environment.LocalDescriptors(8, True)
        assert str(comp) == str(eval(repr(comp)))

    def test_ql(self):
        """Check if we can reproduce Steinhardt ql."""
        # These exact parameter values aren't important; they won't necessarily
        # give useful outputs for some of the structures, but that's fine since
        # we just want to check that LocalDescriptors is consistent with
        # Steinhardt.
        num_neighbors = 6
        l_max = 12

        for struct_func in [
            freud.data.UnitCell.sc,
            freud.data.UnitCell.bcc,
            freud.data.UnitCell.fcc,
        ]:
            box, points = struct_func().generate_system((5, 5, 5))

            # In order to be able to access information on which particles are
            # bonded to which ones, we precompute the neighborlist
            lc = freud.locality.AABBQuery(box, points)
            nl = lc.query(
                points, dict(exclude_ii=True, num_neighbors=num_neighbors)
            ).toNeighborList()
            ld = freud.environment.LocalDescriptors(l_max, mode="global")
            ld.compute((box, points), neighbors=nl)

            ql = get_ql(points, ld, nl)

            # Test all allowable values of l.
            for L in range(2, l_max + 1):
                steinhardt = freud.order.Steinhardt(L)
                steinhardt.compute((box, points), neighbors=nl)
                # Some of the calculations done for Steinhardt can be imprecise
                # in cases where there is no symmetry. Since simple cubic
                # should have a 0 ql value in many cases, we need to set high
                # tolerances for those specific cases.
                atol = 1e-3 if struct_func == freud.data.UnitCell.sc else 1e-6
                npt.assert_allclose(
                    steinhardt.particle_order,
                    ql[:, L],
                    atol=atol,
                    err_msg=f"Failed for {struct_func.__name__}, L = {L}",
                )

    def test_ql_weighted(self):
        """Check if we can reproduce Steinhardt ql with bond weights."""
        np.random.seed(0)

        # These exact parameter values aren't important; they won't necessarily
        # give useful outputs for some of the structures, but that's fine since
        # we just want to check that LocalDescriptors is consistent with
        # Steinhardt.
        num_neighbors = 6
        l_max = 12

        for struct_func in [
            freud.data.UnitCell.sc,
            freud.data.UnitCell.bcc,
            freud.data.UnitCell.fcc,
        ]:
            box, points = struct_func().generate_system((5, 5, 5))

            # In order to be able to access information on which particles are
            # bonded to which ones, we precompute the neighborlist
            lc = freud.locality.AABBQuery(box, points)
            nl = lc.query(
                points, dict(exclude_ii=True, num_neighbors=num_neighbors)
            ).toNeighborList()
            ld = freud.environment.LocalDescriptors(l_max, mode="global")
            ld.compute((box, points), neighbors=nl)

            # Generate random weights for each bond
            nl = freud.locality.NeighborList.from_arrays(
                len(points),
                len(points),
                nl.query_point_indices,
                nl.point_indices,
                nl.distances,
                np.random.rand(len(nl.weights)),
            )

            ql = get_ql(points, ld, nl, True)

            # Test all allowable values of l.
            for L in range(2, l_max + 1):
                steinhardt = freud.order.Steinhardt(L, weighted=True)
                steinhardt.compute((box, points), neighbors=nl)
                # Some of the calculations done for Steinhardt can be imprecise
                # in cases where there is no symmetry. Since simple cubic
                # should have a 0 ql value in many cases, we need to set high
                # tolerances for those specific cases.
                atol = 1e-3 if struct_func == freud.data.UnitCell.sc else 1e-5
                npt.assert_allclose(
                    steinhardt.particle_order,
                    ql[:, L],
                    atol=atol,
                    err_msg=f"Failed for {struct_func.__name__}, L = {L}",
                )

    def test_wl(self):
        """Check if we can reproduce Steinhardt wl."""
        # These exact parameter values aren't important; they won't necessarily
        # give useful outputs for some of the structures, but that's fine since
        # we just want to check that LocalDescriptors is consistent with
        # Steinhardt.
        num_neighbors = 6
        l_max = 12

        for struct_func in [
            freud.data.UnitCell.sc,
            freud.data.UnitCell.bcc,
            freud.data.UnitCell.fcc,
        ]:
            box, points = struct_func().generate_system((5, 5, 5))

            # In order to be able to access information on which particles are
            # bonded to which ones, we precompute the neighborlist
            lc = freud.locality.AABBQuery(box, points)
            nl = lc.query(
                points, dict(exclude_ii=True, num_neighbors=num_neighbors)
            ).toNeighborList()
            ld = freud.environment.LocalDescriptors(l_max, mode="global")
            ld.compute((box, points), neighbors=nl)

            wl = get_wl(points, ld, nl)

            # Test all allowable values of l.
            for L in range(2, l_max + 1):
                steinhardt = freud.order.Steinhardt(L, wl=True)
                steinhardt.compute((box, points), neighbors=nl)
                npt.assert_array_almost_equal(steinhardt.particle_order, wl[:, L])

    def test_ld(self):
        """Verify the behavior of LocalDescriptors by explicitly calculating
        spherical harmonics manually and verifying them."""
        sph_harm = pytest.importorskip("scipy.special").sph_harm

        atol = 1e-5
        L = 8
        N = 100
        box, points = freud.data.make_random_system(L, N)

        num_neighbors = 1
        l_max = 2

        # We want to provide the NeighborList ourselves since we need to use it
        # again later anyway.
        lc = freud.locality.AABBQuery(box, points)
        nl = lc.query(
            points, dict(exclude_ii=True, num_neighbors=num_neighbors)
        ).toNeighborList()

        ld = freud.environment.LocalDescriptors(l_max, mode="global")
        ld.compute((box, points), neighbors=nl)

        # Loop over the sphs and compute them explicitly.
        for idx, (i, j) in enumerate(nl):
            bond = box.wrap(points[j] - points[i])
            r = np.linalg.norm(bond)
            theta = np.arccos(bond[2] / r)
            phi = np.arctan2(bond[1], bond[0])

            count = 0
            for l in range(l_max + 1):
                for m in range(l + 1):
                    # Explicitly calculate the spherical harmonic with scipy
                    # and check the output.  Arg order is theta, phi for scipy,
                    # but we need to pass the swapped angles because it uses
                    # the opposite convention from fsph (which LocalDescriptors
                    # uses internally).
                    scipy_val = sph_harm(m, l, phi, theta)
                    ld_val = (-1) ** abs(m) * ld.sph[idx, count]
                    assert np.isclose(scipy_val, ld_val, atol=atol), (
                        "Failed for l={}, m={}, x={}, y = {}" "\ntheta={}, phi={}"
                    ).format(l, m, scipy_val, ld_val, theta, phi)
                    count += 1

                for neg_m in range(1, l + 1):
                    m = -neg_m
                    scipy_val = sph_harm(m, l, phi, theta)
                    ld_val = ld.sph[idx, count]
                    assert np.isclose(scipy_val, ld_val, atol=atol), (
                        "Failed for l={}, m={}, x={}, y = {}" "\ntheta={}, phi={}"
                    ).format(l, m, scipy_val, ld_val, theta, phi)
                    count += 1

    def test_query_point_ne_points(self):
        """Verify the behavior of LocalDescriptors by explicitly calculating
        spherical harmonics manually and verifying them."""
        sph_harm = pytest.importorskip("scipy.special").sph_harm

        atol = 1e-5
        L = 8
        N = 100
        box, points = freud.data.make_random_system(L, N)
        query_points = np.random.rand(N, 3) * L - L / 2

        num_neighbors = 1
        l_max = 2

        # We want to provide the NeighborList ourselves since we need to use it
        # again later anyway.
        lc = freud.locality.AABBQuery(box, points)
        nl = lc.query(
            points, dict(exclude_ii=True, num_neighbors=num_neighbors)
        ).toNeighborList()

        ld = freud.environment.LocalDescriptors(l_max, mode="global")
        ld.compute((box, points), query_points, neighbors=nl)

        # Loop over the sphs and compute them explicitly.
        for idx, (i, j) in enumerate(nl):
            bond = box.wrap(points[j] - query_points[i])
            r = np.linalg.norm(bond)
            theta = np.arccos(bond[2] / r)
            phi = np.arctan2(bond[1], bond[0])

            count = 0
            for l in range(l_max + 1):
                for m in range(l + 1):
                    # Explicitly calculate the spherical harmonic with scipy
                    # and check the output.  Arg order is theta, phi for scipy,
                    # but we need to pass the swapped angles because it uses
                    # the opposite convention from fsph (which LocalDescriptors
                    # uses internally).
                    scipy_val = sph_harm(m, l, phi, theta)
                    ld_val = (-1) ** abs(m) * ld.sph[idx, count]
                    assert np.isclose(scipy_val, ld_val, atol=atol), (
                        "Failed for l={}, m={}, x={}, y = {}" "\ntheta={}, phi={}"
                    ).format(l, m, scipy_val, ld_val, theta, phi)
                    count += 1

                for neg_m in range(1, l + 1):
                    m = -neg_m
                    scipy_val = sph_harm(m, l, phi, theta)
                    ld_val = ld.sph[idx, count]
                    assert np.isclose(scipy_val, ld_val, atol=atol), (
                        "Failed for l={}, m={}, x={}, y = {}" "\ntheta={}, phi={}"
                    ).format(l, m, scipy_val, ld_val, theta, phi)
                    count += 1

    def test_no_neighbors(self):
        l_max = 8
        box = freud.box.Box.cube(10)
        points = [(0, 0, 0)]

        ld = freud.environment.LocalDescriptors(l_max)
        ld.compute((box, points), neighbors={"r_max": 1.25})
        assert ld.num_sphs == 0
        assert len(ld.sph) == 0
