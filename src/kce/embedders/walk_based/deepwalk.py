from networkx import Graph
import numpy as np
import multiprocessing as mp
from itertools import repeat
import time
from gensim.models import Word2Vec

from .walks import generate_rw
from kce.embedders.embedder import Embedder


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
        pool = mp.Pool()
        nodes = list(graph) * self.n_walks_

        res = pool.starmap_async(func=generate_rw,
                                 iterable=zip(repeat(graph),
                                              nodes,
                                              repeat(self.walk_length_)))
        pool.close()
        walks = res.get()
        return walks

    def embed(self, graph: Graph):
        # Generate random walks
        start_rw_gen = time.time()
        walks = self._generate_walks(graph)
        np.random.shuffle(walks)
        end_rw_gen = time.time()

        # Compute the embedding by training Word2Vec
        wv = self._skip_gram(walks)
        end_embed_train_time = time.time()

        return {
            "n_walks": len(walks),
            "vectors": wv.vectors,
            "node_index": wv.index2word,
            "rw_gen_time": end_rw_gen - start_rw_gen,
            "embed_train_time": end_embed_train_time - end_rw_gen
        }
