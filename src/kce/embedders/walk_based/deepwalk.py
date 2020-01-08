from networkx import Graph
import numpy as np
import multiprocessing as mp
from itertools import repeat
import time
from gensim.models import Word2Vec

from ...utils import timeit
from .walks import generate_rw
from kce.embedders.embedder import Embedder


class DeepWalk(Embedder):

    def __init__(self, out_dim, n_walks, walk_length, win_size, *args, **kwargs):
        super().__init__(out_dim, *args, **kwargs)
        self.n_generated_walks = None
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.win_size = win_size

    def get_attributes(self):
        return {
            **super().get_attributes(),
            "n_walks": self.n_walks,
            "walk_length": self.walk_length,
            "window_size": self.win_size,
            "n_generated_walks": self.n_generated_walks
        }

    @timeit(var_name="skip_gram")
    def _skip_gram(self, walks):
        model = Word2Vec(walks,
                         size=self.out_dim_,
                         window=self.win_size,
                         min_count=0, sg=1, hs=1)
        return model.wv

    @timeit(var_name="generate_walks")
    def _generate_walks(self, graph: Graph):
        nodes = list(graph) * self.n_walks
        with mp.Pool() as pool:
            res = pool.starmap_async(func=generate_rw,
                                     iterable=zip(repeat(graph),
                                                  nodes,
                                                  repeat(self.walk_length)))
            walks = res.get()
        return walks

    def fit(self, graph: Graph, **kwargs):
        # Generate random walks
        walks = self._generate_walks(graph)
        np.random.shuffle(walks)

        # Compute the embedding by training Word2Vec
        wv = self._skip_gram(walks)

        self.n_generated_walks = len(walks)
        self.embeddings = wv.vectors
        self.id2node = list(wv.index2word)
        self.node2id = {word: index for index, word in enumerate(wv.index2word)}

        return self
