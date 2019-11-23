from networkx import Graph, core_number
import numpy as np
from gensim.models import Word2Vec

from .walks import generate_rw
from ..embedder import Embedder
from .deepwalk import DeepWalk


class CoreWalkLinear(DeepWalk):
    def __init__(self, coef=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coef_ = coef

    def _generate_walks(self, graph: Graph):
        core_numbers = core_number(graph)
        max_k = max(core_number.values())
        walks = []
        for node in graph:
            k = core_numbers[node]
            n_walks = max(int(self.coef_ * (k - max_k) + self.n_walks_), 1)
            for i in range(n_walks):
                walks.append(generate_rw(graph, node, self.walk_length_))
        return walks


class CoreWalkPower(DeepWalk):
    def __init__(self, pow=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pow_ = pow

    def _generate_walks(self, graph: Graph):
        core_numbers = core_number(graph)
        max_k = max(core_number.values())
        walks = []
        for node in graph:
            k = core_numbers[node]
            n_walks = max(int(self.n_walks_ * (k/max_k)**self.pow_), 1)
            for i in range(n_walks):
                walks.append(generate_rw(graph, node, self.walk_length_))
        return walks


class CoreWalkSigmoid(DeepWalk):
    def __init__(self, pow=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pow_ = pow

    def sigmoid_(self, x):
        return 1/(1+np.exp(-x))

    def _generate_walks(self, graph: Graph):
        core_numbers = core_number(graph)
        max_k = max(core_number.values())
        walks = []
        for node in graph:
            k = core_numbers[node]
            n_walks = max(
                int(self.n_walks_ * self.sigmoid_(-10*(k - max_k/2)/max_k)),
                1
            )
            for i in range(n_walks):
                walks.append(generate_rw(graph, node, self.walk_length_))
        return walks
