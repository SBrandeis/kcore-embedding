import networkx as nx
import numpy as np


def generate_rw(graph: nx.Graph, node, len_walk):
    walk = [node]

    while len(walk) < len_walk:
        node_neighbours = [n for n in graph[node]]
        weights = []
        norm = 0
        if not node_neighbours:
            return [node]*len_walk
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
