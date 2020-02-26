import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.base import BaseEstimator
from kce.embedders.embedder import Embedder
from copy import deepcopy
from kce.utils import link_pred_train_test_split


def pre_process(graph: nx.Graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph = nx.k_core(graph, k=1)
    return graph


def node_classification_pipeline(graph: nx.Graph, embeddings: np.ndarray, id2node: list, node2id: list, classifier: BaseEstimator,
                                 **kwargs) -> dict:
    test_size = kwargs["test_size"]

    node_vectors = embeddings
    labels = np.array([graph.nodes[word]["community"] for word in id2node])

    node_vectors_train, node_vectors_test, labels_train, labels_test = train_test_split(node_vectors, labels,
                                                                                        test_size=test_size)

    classifier.fit(node_vectors_train, labels_train)
    y_pred = classifier.predict(node_vectors_test)
    y_true = labels_test

    return {
        "micro_f1": f1_score(y_true=y_true, y_pred=y_pred, average="micro"),
        "macro_f1": f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    }


def link_prediction_pipeline(graph: nx.Graph, embeddings: np.array, id2node: list, node2id: list, classifier: BaseEstimator,
                            **kwargs) -> dict:
    non_edges_train, non_edges_test, edges_train, edges_test = kwargs["non_edges_train"], kwargs["non_edges_test"],\
                                                               kwargs["edges_train"], kwargs["edges_test"]
    X_train, X_test, Y_train, Y_test = link_pred_train_test_split(embeddings, node2id, **kwargs)

    # Classify
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    y_true = Y_test

    return {
        "micro_f1": f1_score(y_true=y_true, y_pred=y_pred, average="micro"),
        "macro_f1": f1_score(y_true=y_true, y_pred=y_pred, average="macro"),
        "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred)
    }

def link_prediction_pipeline_old(graph, embedder: Embedder, classifier, embed_kwargs=None, cut_ratio=.5, test_size=.6):

    # Select edges to cut
    index_cut = np.random.choice(graph.number_of_edges(), size=int(graph.number_of_edges()*cut_ratio), replace=False)
    edges = np.array(list(graph.edges))
    nb_edges = edges.shape[0]
    non_edges = np.array(list(nx.non_edges(graph)))
    edges_cut = edges[index_cut]

    # Remove those edges from graph
    _graph = deepcopy(graph)
    [_graph.remove_edge(u, v) for u, v in edges_cut]

    # Create embedding
    if not embed_kwargs:
        embed_kwargs = {}
    embedder.fit(_graph, **embed_kwargs)

    # Create train set : select pairs of nodes not connected
    index_neg = np.random.choice(non_edges.shape[0], size=nb_edges)
    index_train_neg, index_test_neg = index_neg[:-int(nb_edges*test_size)], index_neg[-int(nb_edges*test_size):]
    non_edges_train, non_edges_test = non_edges[index_train_neg], non_edges[index_test_neg]
    index_pos = np.arange(nb_edges)
    np.random.shuffle(index_pos)
    index_train_pos, index_test_pos = index_pos[:-int(nb_edges*test_size)], index_pos[-int(nb_edges*test_size):]
    edges_train, edges_test = edges[index_train_pos], edges[index_test_pos]

    # Retreive corresponding embedding a create train test sets
    X_neg_train = np.array(
        [np.concatenate((embedder.embeddings[embedder.node2id[u]], embedder.embeddings[embedder.node2id[v]])) for u, v
         in non_edges_train])
    X_neg_test = np.array(
        [np.concatenate((embedder.embeddings[embedder.node2id[u]], embedder.embeddings[embedder.node2id[v]])) for u, v
         in non_edges_test])
    X_pos_train = np.array(
        [np.concatenate((embedder.embeddings[embedder.node2id[u]], embedder.embeddings[embedder.node2id[v]])) for u, v
         in edges_train])
    X_pos_test = np.array(
        [np.concatenate((embedder.embeddings[embedder.node2id[u]], embedder.embeddings[embedder.node2id[v]])) for u, v
         in edges_test])
    X_train, X_test = np.concatenate([X_neg_train, X_pos_train]), np.concatenate([X_neg_test, X_pos_test])
    Y_train, Y_test = np.concatenate([np.zeros(X_neg_train.shape[0]), np.ones(X_pos_train.shape[0])]), np.concatenate([np.zeros(X_neg_test.shape[0]), np.ones(X_pos_test.shape[0])])


    # Classify
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    y_true = Y_test

    return {
        **embedder.get_attributes(),
        "micro_f1": f1_score(y_true=y_true, y_pred=y_pred, average="micro"),
        "macro_f1": f1_score(y_true=y_true, y_pred=y_pred, average="macro"),
        "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred)
    }
