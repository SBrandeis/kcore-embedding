from networkx import Graph, core_number
import numpy as np
from gensim.models import Word2Vec

from .walks import generate_rw
from ..embedder import Embedder
from .deepwalk import DeepWalk


class CoreWalk(DeepWalk):

    def _n_walks(self, k, k_max):
        raise NotImplementedError

    def _generate_walks(self, graph: Graph):
        core_numbers = core_number(graph)
        k_max = max(core_numbers.values())
        k_n_walks = [self._n_walks(k, k_max) for k in range(1, k_max + 1)]
        walks = []
        for node in graph:
            k = core_numbers[node]
            n_walks = k_n_walks[k-1]
            for i in range(n_walks):
                walks.append(generate_rw(graph, node, self.walk_length_))
        return walks


class CoreWalkLinear(CoreWalk):

    def __init__(self, offset=0, n_min=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset_ = offset
        self.n_min_ = n_min

    def _n_walks(self, k, k_max):
        return max(int((self.n_walks_ - self.offset_) * (k / k_max) + self.offset_), self.n_min_)


class CoreWalkPower(CoreWalk):
    def __init__(self, pow=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pow_ = pow

    def _n_walks(self, k, k_max):
        return max(int(self.n_walks_ * np.power(k / k_max, self.pow_)), 1)


class CoreWalkSigmoid(CoreWalk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _n_walks(self, k, k_max):
        return max(
                int(self.n_walks_ * self._sigmoid(10*(k - k_max/2)/k_max)),
                1
            )
