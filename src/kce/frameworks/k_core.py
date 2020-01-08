from .framework import Framework, timeit
from networkx import Graph
import numpy as np
from scipy import sparse
import networkx as nx


class KCore(Framework):

    def __init__(self, embedder, **kwargs):
        self.unreachable_nodes = None
        super().__init__(embedder, **kwargs)

    @timeit(var_name="k_core")
    def _k_core_decomposition(self, graph, core_index):
        return nx.k_core(G=graph, k=core_index)

    @timeit(var_name="embed")
    def _embed(self, sub_graph):
        self.embedder_.fit(sub_graph)
        return {
            "embeddings": self.embedder_.embeddings,
            "node2id": self.embedder_.node2id,
            "id2node": self.embedder_.id2node
        }

    def _get_reachable_subgraph(self, graph, from_graph):
        return graph.subgraph([node for node in graph
                               if node not in from_graph
                               and node in from_graph.adj])

    @timeit(var_name="propagate")
    def _propagate(self, graph: Graph, embedded_sub_graph: Graph, embeddings: np.array, node2id, id2node, max_itr: int = 20):
        reachable_nodes = self._get_reachable_subgraph(graph, embedded_sub_graph)
        Z1 = embeddings

        while reachable_nodes.order() > 0:
            A2 = nx.to_scipy_sparse_matrix(reachable_nodes)
            A1 = nx.to_scipy_sparse_matrix(graph)[(node for node in embedded_sub_graph)][(node for node in reachable_nodes)]
            norm = A1.T@A2
            A1 = A1 / norm[:, None]
            A2 = A2 / norm[:, None]
            Z2 = sparse.random(reachable_nodes.order(), self.out_dim_)
            for itr in range(max_itr):
                Z2 = A1@Z1 + A2@Z2

            n1 = embedded_sub_graph.order()
            for node in reachable_nodes:
                node2id[node] = n1
                id2node.append(node)
                n1 = n1 + 1
            embeddings.extend(Z2)
            embedded_sub_graph.add_nodes_from(reachable_nodes)
            reachable_nodes = self._get_reachable_subgraph(graph, embedded_sub_graph)

        unreachable_nodes = graph.subgraph([node for node in graph if node not in embedded_sub_graph])
        unreachable_nodes_embeddings = sparse.random(unreachable_nodes.order(), self.out_dim_)
        embeddings.extend(unreachable_nodes_embeddings)

        n1 = embedded_sub_graph.order()
        for node in unreachable_nodes:
            node2id[node] = n1
            id2node.append(node)
            n1 = n1 + 1

        return {
            "embeddings": embeddings,
            "node2id": node2id,
            "id2node": id2node,
            "unreachable_nodes": unreachable_nodes
        }

    def fit(self, graph: Graph, core_index: int = None, **kwargs):
        k_core = self._k_core_decomposition(graph, core_index)
        embedding_results = self._embed(k_core)
        embeddings = embedding_results["embeddings"]
        node2id = embedding_results["node2id"]
        id2node = embedding_results["id2node"]

        embedded_subgraph = graph.subgraph(node2id.keys())
        propagation_results = self._propagate(graph=graph,
                                              embedded_sub_graph=embedded_subgraph,
                                              embeddings=embeddings,
                                              node2id=node2id,
                                              id2node=id2node)

        self.embeddings = propagation_results["embeddings"]
        self.node2id = propagation_results["node2id"]
        self.id2node = propagation_results["id2node"]
        self.unreachable_nodes = propagation_results["unreachable_nodes"]

        return self
