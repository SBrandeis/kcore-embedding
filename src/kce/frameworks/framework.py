from networkx import Graph
from ..embedders import Embedder
import time


class Framework(Embedder):
    def __init__(self, embedder: Embedder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_dim_ = embedder.out_dim_
        self.embedder_ = embedder

    def fit(self, graph: Graph, **kwargs):
        raise NotImplementedError
