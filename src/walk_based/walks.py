import networkx as nx
import numpy as np


def generate_rw(graph: nx.Graph, node, len_walk):
    walk = [node]

    while len(walk) < len_walk:
        node_neighbours = [n for n in graph[node]]
        weights = []
        norm = 0
        for n in node_neighbours:
            edge = graph[n][node]
            if 'weight' in edge:
                weights.append(edge['weight'])
                norm += edge['weight']
            else:
                weights.append(1)
                norm += 1

        weights = np.array(weights) / norm
        walk.append(np.random.choice(a=node_neighbours, p=weights))

    return walk


def node2vec(graph, win_size, out_dim, n_walks, len_walk, p=0.7, q=10):
    """
    Args:
        graph (networkx.Graph): Input graph, with v vertices
        win_size (int): Window size
        out_dim (int): embedding size
        n_walks (int): walks per vertex
        len_walk (int): walk_based length

    Returns:
        np.array: Representation matrix of shape v x d
    """
    nodes = list(graph.nodes)

    walks = []
    for i in range(n_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walks.append(random_walk(graph, node, len_walk, p, q))
    return skip_gram(walks, out_dim=out_dim, win_size=win_size)
