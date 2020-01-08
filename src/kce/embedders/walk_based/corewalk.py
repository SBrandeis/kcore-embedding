from networkx import Graph, core_number
import numpy as np
import multiprocessing as mp
from itertools import repeat

from ...utils import timeit
from .walks import generate_rw
from .deepwalk import DeepWalk


class CoreWalk(DeepWalk):

    def _n_walks(self, k, k_max):
        raise NotImplementedError

    @timeit(var_name="k_core_decomposition")
    def _k_core_dec(self, graph):
        return core_number(graph)

    @timeit(var_name="generate_walks")
    def _generate_walks(self, graph: Graph):
        core_numbers = self._k_core_dec(graph)
        k_max = max(core_numbers.values())
        k_n_walks = [self._n_walks(k, k_max) for k in range(1, k_max + 1)]

        nodes = [node for node in graph for _ in range(k_n_walks[core_numbers[node] - 1])]

        with mp.Pool() as pool:

            res = pool.starmap_async(func=generate_rw,
                                     iterable=zip(repeat(graph),
                                                  nodes,
                                                  repeat(self.walk_length)))
            walks = res.get()
        return walks


class CoreWalkLinear(CoreWalk):

    def __init__(self, offset=0, n_min=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset_ = offset
        self.n_min_ = n_min

    def _n_walks(self, k, k_max):
        return max(int((self.n_walks - self.offset_) * (k / k_max) + self.offset_), self.n_min_)


class CoreWalkPower(CoreWalk):
    def __init__(self, pow=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pow_ = pow

    def _n_walks(self, k, k_max):
        return max(int(self.n_walks * np.power(k / k_max, self.pow_)), 1)


class CoreWalkSigmoid(CoreWalk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _n_walks(self, k, k_max):
        return max(
                int(self.n_walks * self._sigmoid(10 * (k - k_max / 2) / k_max)),
                1
            )
