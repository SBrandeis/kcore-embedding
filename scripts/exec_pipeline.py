"""
This script is an interface to run experiments.
It executes a classification task after having embedded a graph.
Outputs are:
    1. Embedding results (vectors and node to vector mappings)
    2. Metrics regarding the embedding (execution times)
    3. Metrics regarding the classification task (F1 scores)
"""
import argparse
import csv
from datetime import datetime
from importlib import import_module
import json
from kce.evaluate import node_classification_pipeline, link_prediction_pipeline
import networkx as nx
from os import path, mkdir
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm


AVAILABLE_EMBEDDERS = {
    "deepwalk": ("kce.embedders.walk_based.deepwalk", "DeepWalk"),
    "corewalk_linear": ("kce.embedders.walk_based.corewalk", "CoreWalkLinear"),
    "corewalk_power": ("kce.embedders.walk_based.corewalk", "CoreWalkPower"),
    "corewalk_sigmoid": ("kce.embedders.walk_based.corewalk", "CoreWalkSigmoid"),
}


def load_config(cfg_path):
    """
    Loads a .json file as a dict object
    Arguments:
         cfg_path (str): Path to the .json file
    Returns:
        dict: The loaded config, as a dict
    Raises:
        AssertionError: Raised when cfg_path does not exist or is not a path to a .json file
    """
    assert path.exists(cfg_path)
    assert path.isfile(cfg_path)
    assert cfg_path.split('.')[-1] == "json"
    with open(cfg_path, 'r') as fin:
        config = json.load(fin)
    assert isinstance(config, dict)
    return config


def instantiate_classifier(multilabel=False):
    """
    Instantiates a Linear Classifier, wrapping it in a OneVsRestClassifier if multilabel is True

    Arguments:
        multilabel (bool): Whether the task is a MultiLabel classification task
    Returns:
        sklearn.base.BaseEstimator: The instantiated estimator
    """
    reglog = LogisticRegression(C=1, multi_class="ovr", solver="liblinear")
    if multilabel:
        return OneVsRestClassifier(reglog)
    return reglog


"""
CLI ARGUMENTS DECLARATION
"""
parser = argparse.ArgumentParser(description='TODO')

# Optional config file
parser.add_argument('--config', metavar='config', type=str, nargs='?',
                    help='Path to the config.json file')

# Optional flags
parser.add_argument('--input', metavar='input', type=str, nargs='?',
                    help='Path to the .gml file where the graph is stored')
parser.add_argument('--output', metavar='output', type=str, nargs='?',
                    help='Path to the dir where to store the pipeline results')
parser.add_argument('--base', metavar='base', type=str, nargs='?',
                    help='Base embedder to compare perf with. One of: {}'.format(AVAILABLE_EMBEDDERS.keys()))
parser.add_argument('--target', metavar='target', type=str, nargs='?',
                    help='Target embedder whose perfs are compared to the base one. One of: {}'
                    .format(AVAILABLE_EMBEDDERS.keys()))
parser.add_argument('--reps', metavar='reps', type=int, nargs='?',
                    help='Number of times to repeat experience')
parser.add_argument('--tag', metavar='tag', type=str, nargs='?',
                    help='Tag to identify the run (Optional)')

# Embedders' params
parser.add_argument('--base-params', metavar='base_params', type=str,
                    help='Path to the .json file containing params foir the base embedder.')
parser.add_argument('--target-params', metavar='target_params', type=str,
                    help='Path to the .json file containing params for the target embedder.')
parser.add_argument('--link-pred', metavar='link_pred', type=bool, default=False,
                    help='whether to perform Link prediction instead of node classification.')

if __name__ == '__main__':
    # Parse arguments
    args = parser.parse_args()

    cfg_path = args.config
    tag = args.tag or '0'
    link_pred = args.link_pred

    if cfg_path:
        cfg = load_config(cfg_path)

        input_path = cfg["input_path"]
        output_dir = cfg["output_dir"]
        reps = cfg["reps"]
        base_embedder = cfg["base_embedder"]
        target_embedder = cfg["target_embedder"]

    else:
        input_path = args.input
        output_dir = args.output
        reps = args.reps
        base_embedder = args.base
        target_embedder = args.target

    assert path.exists(input_path)
    assert path.exists(output_dir)
    assert path.isfile(input_path)
    assert path.isdir(output_dir)
    assert reps > 0
    assert base_embedder in AVAILABLE_EMBEDDERS
    assert target_embedder in AVAILABLE_EMBEDDERS

    # Create output dir
    input_name = path.split(input_path)[1].split('.')[0]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_path = path.join(output_dir, "{}_{}_{}_{}_{}".format(input_name, base_embedder, target_embedder, now, tag))
    mkdir(output_path)
    mkdir(output_path + "/embeddings")

    # Load embedders' classes and params
    base_module = AVAILABLE_EMBEDDERS[base_embedder]
    base_class = getattr(import_module(base_module[0]), base_module[1])
    target_module = AVAILABLE_EMBEDDERS[target_embedder]
    target_class = getattr(import_module(target_module[0]), target_module[1])

    base_params_path = args.base_params
    target_params_path = args.target_params
    base_params = load_config(base_params_path)
    target_params = load_config(target_params_path)

    with open(path.join(output_path, "base_params.json"), "w+") as fout:
        json.dump(base_params, fout)
    with open(path.join(output_path, "target_params.json"), "w+") as fout:
        json.dump(target_params, fout)
    with open(path.join(output_path, "script_args.json"), "w+") as fout:
        json.dump({
            "input_path": input_path,
            "output_dir": output_dir,
            "reps": reps,
            "base_embedder": base_embedder,
            "target_embedder": target_embedder,
            "link_pred": link_pred
        }, fout)

    pipeline = link_prediction_pipeline if link_pred else node_classification_pipeline

    # Import and preprocess the graph
    G: nx.Graph = nx.read_gml(input_path)
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
    G = nx.k_core(G, 1)  # Remove orphan nodes

    multilabel = isinstance(list(G.nodes(data="community"))[0][1], list)

    base_metrics = []
    base_embeddings = []
    target_metrics = []
    target_embeddings = []

    try:
        for r in tqdm(range(reps)):
            # Instantiate Embedders
            base = base_class(**base_params)
            target = target_class(**target_params)

            # Execute classification pipeline
            res_base = pipeline(graph=G,
                                embedder=base,
                                classifier=instantiate_classifier(multilabel))

            with open(path.join(output_path, "embeddings", "embeddings_base_{}.pkl".format(r)), "wb+") as fout:
                pickle.dump({
                    "embeddings": res_base.pop("embeddings"),
                    "node2id": res_base.pop("node2id"),
                    "id2node": res_base.pop("id2node")
                }, fout)
            base_metrics.append(res_base)

            res_target = pipeline(graph=G,
                                  embedder=target,
                                  classifier=instantiate_classifier(multilabel))

            with open(path.join(output_path, "embeddings", "embeddings_target_{}.pkl".format(r)), "wb+") as fout:
                pickle.dump({
                    "embeddings": res_target.pop("embeddings"),
                    "node2id": res_target.pop("node2id"),
                    "id2node": res_target.pop("id2node")
                }, fout)
            target_metrics.append(res_target)

    finally:
        # Write metrics as .csv
        fieldnames = base_metrics[0].keys()
        with open(path.join(output_path, 'base_metrics.csv'), 'w+') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(base_metrics)

        fieldnames = target_metrics[0].keys()
        with open(path.join(output_path, 'target_metrics.csv'), 'w+') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(target_metrics)
