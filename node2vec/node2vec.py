import networkx as nx
import numpy as np
from gensim.models import Word2Vec


def random_walk(graph: nx.Graph, node, len_walk, p, q):
    assert p
    assert q
    walk = [node, np.random.choice(a=list(graph.neighbors(node)))]

    while len(walk) < len_walk:
        prev = walk[-2]
        node = walk[-1]
        node_neighbours = list(graph.neighbors(node))
        prev_neighbours = list(graph.neighbors(prev))

        weights = []

        for n in node_neighbours:
            if n == prev:
                weights.append(1/p)
            elif n in prev_neighbours:
                weights.append(1)
            else:
                weights.append(1/q)

        norm = sum(weights)
        weights = np.array(weights) / norm
        walk.append(np.random.choice(a=node_neighbours, p=weights))
    return walk


def skip_gram(walks, out_dim, win_size):
    model = Word2Vec(walks, size=out_dim, window=win_size, min_count=0, sg=1, hs=1)
    return model.wv


def node2vec(graph, win_size, out_dim, n_walks, len_walk, p=0.7, q=10):
    """
    Args:
        graph (networkx.Graph): Input graph, with v vertices
        win_size (int): Window size
        out_dim (int): embedding size
        n_walks (int): walks per vertex
        len_walk (int): walk length

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
