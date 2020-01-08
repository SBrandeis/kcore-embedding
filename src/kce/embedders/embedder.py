from networkx import Graph
import numpy as np


class Embedder:

    def __init__(self, out_dim: int, *args, **kwargs):
        """
        Spawns Embedder instance
        :param out_dim: Embedding space dimension
        :param args:
        :param kwargs:
        """
        self.out_dim_ = out_dim
        self.node2id = None
        self.id2node = None
        self.embeddings = None
        self.times = {}

    def fit(self, graph: Graph, **kwargs):
        """
        Computes embeddings for the given graph.

        :param graph: The input graph
        :type graph: networkx.Graph
        :return: self
        """
        raise NotImplementedError

    def get_attributes(self):
        self.times["total"] = sum(self.times.values())
        return {
            "out_dim": self.out_dim_,
            "exec_times": self.times,
            "embeddings": self.embeddings,
            "node2id": self.node2id,
            "id2node": self.id2node
        }

    def transform(self, graph: Graph, nodelist=None):
        if not self.embeddings and not self.node2id:
            raise Exception("Not fitted yet")

        if nodelist:
            return np.array([
                self.embeddings[self.node2id[node]] for node in nodelist if node in graph
            ])
        else:
            return np.array([
                self.embeddings[self.node2id[node]] for node in graph
            ])

    def fit_transform(self, graph: Graph, nodelist=None, **kwargs):
        self.fit(graph, **kwargs)
        return self.transform(graph, nodelist)
