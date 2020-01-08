import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from kce.embedders.embedder import Embedder


def pre_process(graph: nx.Graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph = nx.k_core(graph, k=1)
    return graph


def node_classification_pipeline(graph, embedder: Embedder, classifier, test_size=0.6):
    embedding_results = embedder.embed(graph)

    X = embedding_results["vectors"]
    Y = np.array([graph.nodes[word]["community"] for word in embedding_results["node_index"]])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    y_true = Y_test

    return {
        **embedding_results,
        "micro_f1": f1_score(y_true=y_true, y_pred=y_pred, average="micro"),
        "macro_f1": f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    }
