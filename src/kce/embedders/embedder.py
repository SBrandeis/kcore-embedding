from networkx import Graph
from numpy import ndarray


class Embedder:

    def __init__(self, out_dim: int, *args, **kwargs):
        self.out_dim_ = out_dim

    def embed(self, graph: Graph) -> dict:
        """
        Computes and returns embedding for the input graph.

        :param graph: The input graph
        :type graph: networkx.Graph
        :return: A dictionary containing the embedding results.
                The embedding is stored under key "vectors".
                The corresponding nodes are stored in key "node_index".
        :rtype: dict
        """
        raise NotImplementedError
