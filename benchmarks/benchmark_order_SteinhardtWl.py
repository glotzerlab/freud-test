import numpy as np
import freud
from benchmark import Benchmark
from benchmarker import run_benchmarks


class BenchmarkOrderSteinhardtWl(Benchmark):
    def __init__(self, L, r_max, sph_l):
        self.L = L
        self.r_max = r_max
        self.sph_l = sph_l

    def bench_setup(self, N):
        self.box = freud.box.Box.cube(self.L)
        seed = 0
        np.random.seed(seed)
        self.points = np.asarray(np.random.uniform(-self.L/2, self.L/2,
                                                   (N, 3)),
                                 dtype=np.float32)
        self.lql = freud.order.Steinhardt(self.sph_l, wl=True)

    def bench_run(self, N):
        self.lql.compute((self.box, self.points),
                         neighbors={'r_max': self.r_max})


def run():
    Ns = [100, 500, 1000, 5000]
    number = 100
    name = 'freud.order.SteinhardtWl'

    kwargs = {"L": 10,
              "r_max": 1.5,
              "sph_l": 6}

    return run_benchmarks(name, Ns, number, BenchmarkOrderSteinhardtWl,
                          **kwargs)


if __name__ == '__main__':
    run()
