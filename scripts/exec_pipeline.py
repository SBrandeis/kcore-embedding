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
import json
import logging
import pickle
from datetime import datetime
from importlib import import_module
from os import path, mkdir, makedirs

import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm

from kce.evaluate import node_classification_pipeline, link_prediction_pipeline
from kce.utils import preprocess, downstream_specific_preprocessing

import numpy as np
import random

def set_reproductible():
    np.random.seed(0)
    random.seed(0)

AVAILABLE_EMBEDDERS = {
    "deepwalk": ("kce.embedders.walk_based.deepwalk", "DeepWalk"),
    "corewalk_linear": ("kce.embedders.walk_based.corewalk", "CoreWalkLinear"),
    "corewalk_power": ("kce.embedders.walk_based.corewalk", "CoreWalkPower"),
    "corewalk_sigmoid": ("kce.embedders.walk_based.corewalk", "CoreWalkSigmoid"),
    "k_core": ("kce.frameworks.k_core", "KCore")
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
    reglog = LogisticRegression(C=1, multi_class="ovr", solver="liblinear", random_state=0)
    if multilabel:
        return OneVsRestClassifier(reglog)
    return reglog


def instantiate_embedder(name, params, sub_embedder_name=None, sub_embedder_params=None):
    embedder_module = AVAILABLE_EMBEDDERS[name]
    if "kce.frameworks" in embedder_module[0]:
        assert isinstance(sub_embedder_name, str)
        assert isinstance(sub_embedder_params, dict)
        sub_embedder = instantiate_embedder(name=sub_embedder_name, params=sub_embedder_params)
        params["embedder"] = sub_embedder
    embedder_class = getattr(import_module(embedder_module[0]), embedder_module[1])
    embedder = embedder_class(**params)
    return embedder

def main(args):
    set_reproductible()
    error_code = 0
    cfg_path = args.config
    tag = args.tag or '0'
    verbose = args.verbose or 1
    if verbose <= 0:
        log_level = logging.ERROR
    elif verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s:\t%(message)s', level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    logger.info("Loading config...")
    if cfg_path:
        cfg = load_config(cfg_path)
        input_path = cfg["input_path"]
        output_dir = cfg["output_dir"]
        reps = cfg["reps"]
        embedder_name = cfg["embedder"]
        sub_embedder_name = cfg.get("sub_embedder", None)
        link_pred = cfg["link_pred"]
        save_embedding = cfg["save_embedding"]
        downstream_task_args = cfg.get("downstream_task_args", {})

    else:
        input_path = args.input
        output_dir = args.output
        reps = args.reps
        embedder_name = args.embedder
        sub_embedder_name = args.sub_embedder or None
        link_pred = args.link_pred
        downstream_task_args = {}
        save_embedding = args.save_embedding
        graph_non_edges = args.graph_non_edges

    if not path.exists(output_dir):
        makedirs(output_dir)

    assert path.exists(input_path), "Unable to retreive data path. Refer to doc"
    assert path.exists(output_dir)
    assert path.isfile(input_path)
    assert path.isdir(output_dir)
    assert reps > 0
    assert embedder_name in AVAILABLE_EMBEDDERS
    assert sub_embedder_name is None or sub_embedder_name in AVAILABLE_EMBEDDERS
    logger.debug("Config succesully loaded")

    # Create output dir
    logger.debug("Creating output directory...")
    input_name = path.split(input_path)[1].split('.')[0]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_path = path.join(output_dir, "{}_{}_{}_{}".format(input_name, embedder_name, now, tag))
    mkdir(output_path)
    mkdir(output_path + "/embeddings")
    logger.debug("Output directory successfully created")

    # Load embedders' params
    logger.debug("Loading embedders' configs...")
    embedder_params_path = args.params
    sub_embedder_params_path = args.sub_embedder_params
    embedder_params = load_config(embedder_params_path)
    if sub_embedder_params_path is not None:
        sub_embedder_params = load_config(sub_embedder_params_path)
    else:
        sub_embedder_params = {}
    logger.debug("Embedders' configs successfully loaded")

    logger.debug("Dumping configs in output directory...")
    with open(path.join(output_path, "embedder_params.json"), "w+") as fout:
        json.dump(embedder_params, fout)
    if sub_embedder_params:
        with open(path.join(output_path, "sub_embedder_params.json"), "w+") as fout:
            json.dump(sub_embedder_params, fout)
    with open(path.join(output_path, "script_args.json"), "w+") as fout:
        json.dump({
            "input_path": input_path,
            "output_dir": output_dir,
            "reps": reps,
            "embedder": embedder_name,
            "sub_embedder": sub_embedder_name,
            "link_pred": link_pred
        }, fout)
    logger.debug("Successfully dumped configs in output directory")

    train_params = embedder_params.pop("train", None)

    downstream_task, downstream_task_name = (link_prediction_pipeline, "link_prediction") if link_pred else \
        (node_classification_pipeline, "node_classification")


    # Import and preprocess the graph
    logger.info("Loading graph...")
    graph: nx.Graph = nx.read_gml(input_path)

    logger.debug("Preprocessing graph")
    isolated, selfloop = preprocess(graph) # Preprocessing inplace to reduce memory usage
    logger.debug("Preprocessing: removing {} isolated nodes and {} selfloop edges".format(len(list(isolated)), len(list(selfloop))))
    logger.debug("Graph successfully loaded and preprocessed")

    multilabel = isinstance(list(graph.nodes(data="community"))[0][1], list)

    metrics = []

    try:
        logger.debug("Starting {} downstream_task for graph: {}".format("missing edge detection"
                                                                 if link_pred
                                                                 else "node classification",
                                                                 input_name))

        if verbose >= 1:
            reps_iter = tqdm(range(reps))
        else:
            reps_iter = range(reps)
        for r in reps_iter:
            embedder = instantiate_embedder(name=embedder_name, params=embedder_params,
                                            sub_embedder_name=sub_embedder_name, sub_embedder_params=sub_embedder_params)

            logger.info("Task specific preprocessing.. ")
            graph_, preprocessing_dict = downstream_specific_preprocessing(graph,
                                                                           downstream_task_name=downstream_task_name,
                                                                           **downstream_task_args)
            downstream_task_args.update(**preprocessing_dict)

            logger.info("Fitting embedder...")
            embedder.fit(graph_, **train_params)
            logger.debug("Done.")
            # Create embedding

            embedding_results = embedder.get_attributes()
            embeddings = embedding_results.pop("embeddings")
            id2node = embedding_results.pop("id2node")
            node2id = embedding_results.pop("node2id")

            if save_embedding:
                logger.debug("Dumping embeddings...")
                with open(path.join(output_path, "embeddings", "embeddings_{}.pkl".format(r)), "wb+") as fout:
                    pickle.dump({
                        "embeddings": embeddings,
                        "node2id": node2id,
                        "id2node": id2node
                    }, fout)
                logger.debug("Done.")

            logger.info("Classify with base embeddings...")
            embedding_results.update(
                downstream_task(graph=graph, embeddings=embeddings, id2node=id2node, node2id=node2id,
                                classifier=instantiate_classifier(multilabel),
                                **downstream_task_args)
            )
            logger.debug("Done.")
            metrics.append(embedding_results)

            logger.debug("Cleaning up...")
            del embeddings
            del id2node
            del node2id
            del embedder

    except Exception as e:
        logger.error("Exception raised while executing downstream_task:")
        logger.error(str(e))
        error_code = 1
        pass
    finally:
        logger.info("Saving metrics...")
        # Write metrics as .csv
        fieldnames = metrics[0].keys()
        with open(path.join(output_path, 'base_metrics.csv'), 'w+') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
        logger.debug("Done.")

    return error_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')

    parser.add_argument("--verbose", metavar="verbose", type=int, default=1,
                        help="Verbosity, higher value = more messages.")
    # Optional config file
    parser.add_argument('--config', metavar='config', type=str, nargs='?',
                        help='Path to the config.json file')

    # Optional flags
    parser.add_argument('--input-path', '-i', metavar='input', type=str, nargs='?',
                        help='Path to the .gml file where the graph is stored')
    parser.add_argument('--output-dir', '-o', metavar='output', type=str, nargs='?',
                        help='Path to the dir where to store the pipeline results')
    parser.add_argument('--embedder', '-e', metavar='embedder', type=str, nargs='?',
                        help='Embedder to use. One of: {}'.format(AVAILABLE_EMBEDDERS.keys()))
    parser.add_argument('--sub-embedder', metavar='sub_embedder', type=str, nargs='?',
                        help='Base embedder to use when embedder is a Framework. One of: {}'
                        .format(AVAILABLE_EMBEDDERS.keys()))
    parser.add_argument('--reps', metavar='reps', type=int, nargs='?',
                        help='Number of times to repeat experience')
    parser.add_argument('--tag', metavar='tag', type=str, nargs='?',
                        help='(Optional) Tag to identify the run.')

    # Embedders' params
    parser.add_argument('--params', '-p', metavar='params', type=str,
                        help='Path to the .json file containing params for the embedder.')
    parser.add_argument('--sub-embedder-params', metavar='sub_embedder_params', type=str, nargs='?',
                        help='Path to the .json file containing params for the embedder.')
    parser.add_argument('--link-pred', '-l', metavar='link_pred', type=bool, default=False,
                        help='whether to perform Link prediction instead of node classification.')
    parser.add_argument('--save_embedding', '-s', metavar='save_embedding', type=bool, default=False,
                        help='whether to save embeddings along with downstream task results or not.')
    # Parse arguments
    args = parser.parse_args()
    main(args)
