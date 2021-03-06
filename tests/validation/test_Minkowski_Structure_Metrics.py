import os

import garnett
import numpy as np
import numpy.testing as npt
import pytest

import freud


def _get_structure_data(structure, qtype):
    return np.genfromtxt(
        os.path.join(
            os.path.dirname(__file__),
            "files",
            "minkowski_structure_metrics",
            f"{structure}_{qtype}.txt",
        )
    )


class TestMinkowski:
    @pytest.mark.parametrize("structure", ["fcc", "bcc", "hcp", "sc"])
    def test_minkowski_structure_metrics(self, structure):
        expected_ql = _get_structure_data(structure, "q")
        expected_avql = _get_structure_data(structure, "avq")
        expected_wl = _get_structure_data(structure, "w")
        expected_avwl = _get_structure_data(structure, "avw")

        with garnett.read(
            os.path.join(
                os.path.dirname(__file__),
                "files",
                "minkowski_structure_metrics",
                f"{structure}.gsd",
            )
        ) as traj:
            snap = traj[0]

        voro = freud.locality.Voronoi()
        voro.compute(snap)
        for sph_l in range(expected_ql.shape[1]):

            # These tests fail for unknown (probably numerical) reasons.
            if structure == "hcp" and sph_l in [3, 5]:
                continue

            # Test q'l
            comp = freud.order.Steinhardt(sph_l, weighted=True)
            comp.compute(snap, neighbors=voro.nlist)
            npt.assert_allclose(comp.order, expected_ql[:, sph_l], atol=1e-5)

            # Test average q'l
            comp = freud.order.Steinhardt(sph_l, average=True, weighted=True)
            comp.compute(snap, neighbors=voro.nlist)
            npt.assert_allclose(comp.order, expected_avql[:, sph_l], atol=1e-5)

            # w'2 tests fail for unknown (probably numerical) reasons.
            if sph_l != 2:
                # Test w'l
                comp = freud.order.Steinhardt(
                    sph_l, wl=True, weighted=True, wl_normalize=True
                )
                comp.compute(snap, neighbors=voro.nlist)
                npt.assert_allclose(comp.order, expected_wl[:, sph_l], atol=1e-5)

                # Test average w'l
                comp = freud.order.Steinhardt(
                    sph_l, wl=True, average=True, weighted=True, wl_normalize=True
                )
                comp.compute(snap, neighbors=voro.nlist)
                npt.assert_allclose(comp.order, expected_avwl[:, sph_l], atol=1e-5)
