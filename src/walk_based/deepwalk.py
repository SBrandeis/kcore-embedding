from networkx import Graph
import numpy as np
import time
from gensim.models import Word2Vec

from .walks import generate_rw
from ..embedder import Embedder


class DeepWalk(Embedder):

    def __init__(self, out_dim, n_walks, walk_length, win_size, *args, **kwargs):
        super().__init__(out_dim, *args, **kwargs)
        self.n_walks_ = n_walks
        self.walk_length_ = walk_length
        self.win_size_ = win_size

    def _skip_gram(self, walks):
        model = Word2Vec(walks,
                         size=self.out_dim_,
                         window=self.win_size_,
                         min_count=0, sg=1, hs=1)
        return model.wv

    def _generate_walks(self, graph: Graph):
        walks = []
        for i in range(self.n_walks_):
            for node in graph:
                walks.append(generate_rw(graph, node, self.walk_length_))
        return walks

    def embed(self, graph: Graph):
        # Generate random walks
        start_rw_gen = time.time()
        walks = self._generate_walks(graph)
        np.random.shuffle(walks)
        rw_gen_time = time.time() - start_rw_gen

        # Compute the embedding by training Word2Vec
        start_embed_train = time.time()
        wv = self._skip_gram(walks)
        embed_train_time = time.time() - start_embed_train

        return {
            "n_walks": len(walks),
            "vectors": wv.vectors,
            "node_index": wv.index2word,
            "rw_gen_time": rw_gen_time,
            "embed_train_time": embed_train_time
        }
