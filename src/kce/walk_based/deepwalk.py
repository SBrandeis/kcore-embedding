from networkx import Graph
import numpy as np
import time
from gensim.models import Word2Vec
import multiprocessing as mp

from .walks import generate_rw
from ..embedder import Embedder


class DeepWalk(Embedder):

    def __init__(self, out_dim, n_walks, walk_length, win_size, multiprocessing, *args, **kwargs):
        super().__init__(out_dim, *args, **kwargs)
        self.n_walks_ = n_walks
        self.walk_length_ = walk_length
        self.win_size_ = win_size
        
        self.verbose = verbose
        self.multiprocessing = multiprocessing

        self.generate_walks = self._generate_walks_singleprocessing if not self.multiprocessing else self._generate_walks_multiprocessing

    def _skip_gram(self, walks):
        model = Word2Vec(walks,
                         size=self.out_dim_,
                         window=self.win_size_,
                         min_count=0, sg=1, hs=1)
        return model.wv

    def _generate_walks_singleprocessing(self, graph: Graph):
        walks = []
        for i in range(self.n_walks_):
            for node in graph:
                walks.append(generate_rw(graph, node, self.walk_length_))
        return walks
    
    def _generate_walks_multiprocessing(self, graph:Graph):

        def _random_walk(node, rep, len_walk):
            return [generate_rw(graph, node, len_walk) for _ in range(rep)]

        p = mp.Pool(mp.cpu_count())
        multiprocess_rw = partial(_random_walk, rep=rep, len_walk=self.walk_length_) 
        walks = p.map(multiprocess_rw, [n for n in G.nodes])
        walks = [[node for walk in l for node in walk] for l in walks] # Reshape to (node, len_walk*n_walks)
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
