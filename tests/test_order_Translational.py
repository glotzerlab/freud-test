import itertools

import numpy as np
import numpy.testing as npt
import pytest
import util

import freud
from freud.errors import FreudDeprecationWarning


class TestTranslational:
    def test_simple(self):
        box = freud.box.Box.square(10)

        # make a square grid
        xs = np.linspace(-box.Lx / 2, box.Lx / 2, 10, endpoint=False)
        positions = np.zeros((len(xs) ** 2, 3), dtype=np.float32)
        positions[:, :2] = np.array(list(itertools.product(xs, xs)), dtype=np.float32)

        r_max = 1.1
        n = 4
        with pytest.warns(FreudDeprecationWarning):
            trans = freud.order.Translational(4)
        # Test access
        with pytest.raises(AttributeError):
            trans.particle_order

        test_set = util.make_raw_query_nlist_test_set(
            box, positions, positions, "nearest", r_max, n, True
        )
        for nq, neighbors in test_set:
            trans.compute(nq, neighbors=neighbors)
            # Test access
            trans.particle_order

            npt.assert_allclose(trans.particle_order, 0, atol=1e-6)

    def test_repr(self):
        with pytest.warns(FreudDeprecationWarning):
            trans = freud.order.Translational(4)
            assert str(trans) == str(eval(repr(trans)))

    def test_no_neighbors(self):
        box = freud.box.Box.square(10)
        positions = [(0, 0, 0)]
        trans = freud.order.Translational(4)
        trans.compute((box, positions), neighbors={"r_max": 1.25})
