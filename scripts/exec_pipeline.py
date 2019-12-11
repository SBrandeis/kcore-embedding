import argparse
import csv
from os import path
from kce.walk_based.deepwalk import DeepWalk
from kce.walk_based.corewalk import CoreWalkLinear
from kce.evaluate import node_classification_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
import networkx as nx


def instantiate_classifier(multilabel=False):
    reglog = LogisticRegression(C=1, multi_class="ovr", solver="lbfgs")
    if multilabel:
        return OneVsRestClassifier(reglog)
    return reglog

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('input', metavar='input', type=str,
                    help='Path to the .gml file where the graph is stored')

parser.add_argument('output', metavar='output', type=str,
                    help='Path to the dir where to store the pipeline results')

parser.add_argument('reps', metavar='reps', type=int,
                    help='Number of times to repeat experience')


if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output
    reps = args.reps

    assert path.exists(input_path)
    assert path.exists(output_dir)
    assert path.isfile(input_path)
    assert path.isdir(output_dir)
    assert reps > 0

    G: nx.Graph = nx.read_gml(input_path)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.k_core(G, 1)

    multilabel = isinstance(list(G.nodes(data="community"))[0][1], list)

    params = dict(
        out_dim=150,
        n_walks=15,
        walk_length=25,
        win_size=5,
    )

    deepwalk_results = []
    corewalk_results = []
    try:
        for r in tqdm(range(reps)):
            dw = DeepWalk(**params)
            cw = CoreWalkLinear(**params)

            res_dw = node_classification_pipeline(graph=G,
                                                  embedder=dw,
                                                  classifier=instantiate_classifier(multilabel))
            deepwalk_results.append(res_dw)

            res_cw = node_classification_pipeline(graph=G,
                                                  embedder=cw,
                                                  classifier=instantiate_classifier(multilabel))
            corewalk_results.append(res_cw)
    finally:
        input_name = path.split(input_path)[1].split('.')[0]
        fieldnames = corewalk_results[0].keys()
        with open(path.join(output_dir, input_name + '_cw.csv'), 'w+') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(corewalk_results)

        fieldnames = deepwalk_results[0].keys()
        with open(path.join(output_dir, input_name + '_dw.csv'), 'w+') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(deepwalk_results)
