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
import logging
import networkx as nx
from os import path, mkdir
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm


logger = logging.getLogger()

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
    reglog = LogisticRegression(C=1, multi_class="ovr", solver="liblinear")
    if multilabel:
        return OneVsRestClassifier(reglog)
    return reglog


def instantiate_embedder(name, params, sub_embedder_name=None, sub_embedder_params=None):
    embedder_module = AVAILABLE_EMBEDDERS[name]
    if "kce.frameworks" in embedder_module[0]:
        assert sub_embedder_name is str
        assert sub_embedder_params is dict
        sub_embedder = instantiate_embedder(name=sub_embedder_name, params=sub_embedder_params)
        params["embedder"] = sub_embedder
    embedder_class = getattr(import_module(embedder_module[0]), embedder_module[1])
    embedder = embedder_class(**params)
    return embedder


"""
CLI ARGUMENTS DECLARATION
"""
def main(args):
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
    logger.setLevel(log_level)

    logger.info("Loading config...")
    if cfg_path:
        cfg = load_config(cfg_path)
        input_path = cfg["input_path"]
        output_dir = cfg["output_dir"]
        reps = cfg["reps"]
        base_embedder_name = cfg["base_embedder"]
        target_embedder_name = cfg["target_embedder"]
        link_pred = cfg["link_pred"]

    else:
        input_path = args.input
        output_dir = args.output
        reps = args.reps
        base_embedder_name = args.base
        target_embedder_name = args.target
        link_pred = args.link_pred

    assert path.exists(input_path)
    assert path.exists(output_dir)
    assert path.isfile(input_path)
    assert path.isdir(output_dir)
    assert reps > 0
    assert base_embedder_name in AVAILABLE_EMBEDDERS
    assert target_embedder_name in AVAILABLE_EMBEDDERS
    logger.debug("Config succesully loaded")

    # Create output dir
    logger.debug("Creating output directory...")
    input_name = path.split(input_path)[1].split('.')[0]
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    output_path = path.join(output_dir, "{}_{}_{}_{}_{}".format(input_name, base_embedder_name, target_embedder_name, now, tag))
    mkdir(output_path)
    mkdir(output_path + "/embeddings")
    logger.debug("Output directory successfully created")

    # Load embedders' params
    logger.debug("Loading embedders' configs...")
    base_params_path = args.base_params
    target_params_path = args.target_params
    base_params = load_config(base_params_path)
    target_params = load_config(target_params_path)
    logger.debug("Embedders' configs successfully loaded")

    logger.debug("Dumping configs in output directory...")
    with open(path.join(output_path, "base_params.json"), "w+") as fout:
        json.dump(base_params, fout)
    with open(path.join(output_path, "target_params.json"), "w+") as fout:
        json.dump(target_params, fout)
    with open(path.join(output_path, "script_args.json"), "w+") as fout:
        json.dump({
            "input_path": input_path,
            "output_dir": output_dir,
            "reps": reps,
            "base_embedder": base_embedder_name,
            "target_embedder": target_embedder_name,
            "link_pred": link_pred
        }, fout)
    logger.debug("Successfully dumped configs in output directory")

    base_train_params = base_params.pop("train", None)
    target_train_params = target_params.pop("train", None)

    pipeline = link_prediction_pipeline if link_pred else node_classification_pipeline

    # Import and preprocess the graph
    logger.info("Loading graph...")
    G: nx.Graph = nx.read_gml(input_path)

    logger.debug("Preprocessing graph")
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops
    G = nx.k_core(G, 1)  # Remove orphan nodes
    logger.debug("Graph successfully loaded and preprocessed")

    multilabel = isinstance(list(G.nodes(data="community"))[0][1], list)

    base_metrics = []
    target_metrics = []

    try:
        logger.debug("Starting {} pipeline for graph: {}".format("missing edge detection"
                                                                 if link_pred
                                                                 else "node classification",
                                                                 input_name))

        if verbose >= 1:
            reps_iter = tqdm(range(reps))
        else:
            reps_iter = range(reps)
        for r in reps_iter:
            # BASE EMBEDDER
            base_embedder = instantiate_embedder(name=base_embedder_name, params=base_params)

            logger.info("Fitting base embedder...")
            base_embedder.fit(G, **base_train_params)
            logger.debug("Done.")
            res_base = base_embedder.get_attributes()
            embeddings = res_base.pop("embeddings")
            id2node = res_base.pop("id2node")
            node2id = res_base.pop("node2id")

            logger.debug("Dumping base embeddings...")
            with open(path.join(output_path, "embeddings", "embeddings_base_{}.pkl".format(r)), "wb+") as fout:
                pickle.dump({
                    "embeddings": embeddings,
                    "node2id": node2id,
                    "id2node": id2node
                }, fout)
            logger.debug("Done.")

            logger.info("Classify with base embeddings...")
            res_base.update(
                pipeline(graph=G, embeddings=embeddings, id2node=id2node,
                         classifier=instantiate_classifier(multilabel))
            )
            logger.debug("Done.")
            base_metrics.append(res_base)

            logger.debug("Cleaning up...")
            del embeddings
            del id2node
            del node2id
            del base_embedder

            # TARGET EMBEDDER
            target_embedder = instantiate_embedder(name=target_embedder_name, params=target_params)

            logger.info("Fitting target embedder...")
            target_embedder.fit(G, **target_train_params)
            logger.debug("Done.")
            res_target = target_embedder.get_attributes()
            embeddings = res_target.pop("embeddings")
            id2node = res_target.pop("id2node")
            node2id = res_target.pop("node2id")

            logger.debug("Dumping target embeddings...")
            with open(path.join(output_path, "embeddings", "embeddings_target_{}.pkl".format(r)), "wb+") as fout:
                pickle.dump({
                    "embeddings": embeddings,
                    "node2id": node2id,
                    "id2node": id2node
                }, fout)
            logger.debug("Done.")

            logger.info("Classify with target embeddings...")
            res_target.update(
                pipeline(graph=G, embeddings=embeddings, id2node=id2node,
                         classifier=instantiate_classifier(multilabel))
            )
            logger.debug("Done.")
            target_metrics.append(res_target)

            logger.debug("Cleaning up...")
            del embeddings
            del id2node
            del node2id
            del target_embedder

    except Exception as e:
        logger.error("Exception raised while executing pipeline:")
        logger.error(str(e))
        error_code = 1
        pass
    finally:
        logger.info("Saving metrics...")
        # Write metrics as .csv
        logger.debug("Saving base metrics...")
        fieldnames = base_metrics[0].keys()
        with open(path.join(output_path, 'base_metrics.csv'), 'w+') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(base_metrics)
        logger.debug("Done.")

        logger.debug("Saving target metrics...")
        fieldnames = target_metrics[0].keys()
        with open(path.join(output_path, 'target_metrics.csv'), 'w+') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(target_metrics)
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
    # Parse arguments
    args = parser.parse_args()
    main(args)
