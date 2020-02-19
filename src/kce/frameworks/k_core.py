from .framework import Framework
from networkx import Graph
from kce.utils import timeit
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
        adj_nodes = set().union(*[set(graph.neighbors(node)) for node in list(from_graph)])
        reachable_node = [node for node in graph if node in adj_nodes and node not in from_graph]
        return graph.subgraph(reachable_node)


    @timeit(var_name="propagate")
    def _propagate(self, graph: Graph, embedded_sub_graph: Graph, embeddings: np.array, node2id, id2node, max_itr: int = 20):
        reachable_nodes = self._get_reachable_subgraph(graph, embedded_sub_graph)

        Z1 = embeddings

        sparse_adj = nx.to_scipy_sparse_matrix(graph)

        while reachable_nodes.order() > 0:
            print("Propagating : current embedding size : {}, reachable nodes : {}, total graph size : {}".format(
                len(embedded_sub_graph),
                len(reachable_nodes),
                len(graph)))
            Z1 = embeddings
            reachable_indexes = [i for i, node in enumerate(graph) if node in reachable_nodes]
            embedded_indexes = [i for i, node in enumerate(graph) if node in embedded_sub_graph]
            A1, A2 = sparse_adj[embedded_indexes, :][:, reachable_indexes], sparse_adj[reachable_indexes, :][:, reachable_indexes]
            norm = sparse.hstack([A1.T, A2]).sum(axis=1)
            A1_norm, A2_norm = sparse.csc_matrix(A1/norm.T), sparse.csc_matrix(A2/norm.T)
            Z2 = sparse.csr_matrix(np.random.uniform(-1, 1, size=(reachable_nodes.order(), self.out_dim_)))
            for itr in range(max_itr):
                Z2 = A1_norm.T@Z1 + A2_norm.T@Z2


            n1 = embedded_sub_graph.order()
            for node in reachable_nodes:
                node2id[node] = n1
                id2node.append(node)
                n1 = n1 + 1
            embeddings = np.concatenate([embeddings, Z2], axis=0)
            embedded_sub_graph = graph.subgraph(list(node2id.keys()))
            reachable_nodes = self._get_reachable_subgraph(graph, embedded_sub_graph)

        unreachable_nodes = graph.subgraph([node for node in graph if node not in embedded_sub_graph])
        if unreachable_nodes:
            unreachable_nodes_embeddings = sparse.csr_matrix(np.zeros((unreachable_nodes.order(), self.out_dim_)))
            embeddings = np.concatenate([embeddings, unreachable_nodes_embeddings.todense()], axis=0)

            n1 = embedded_sub_graph.order()
            for node in unreachable_nodes:
                node2id[node] = n1
                id2node.append(node)
                n1 = n1 + 1

        return {
            "embeddings": np.array(embeddings),
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
