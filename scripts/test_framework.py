from tqdm import tqdm
from kce.embedders import DeepWalk
from kce.frameworks import KCore
from kce.evaluate import node_classification_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import argparse
import csv
import networkx as nx
from os import path


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

    dw_results = []
    fw_results = []

    try:
        for r in tqdm(range(reps)):
            dw = DeepWalk(**params)
            fw = KCore(embedder=dw, out_dim=150)

            res_dw = node_classification_pipeline(graph=G,
                                                  embedder=dw,
                                                  classifier=instantiate_classifier(multilabel))
            dw_results.append(res_dw)

            res_fw = node_classification_pipeline(graph=G,
                                                  embedder=fw,
                                                  classifier=instantiate_classifier(multilabel),
                                                  embed_kwargs={"core_index": 2})
            fw_results.append(res_fw)
    finally:
        input_name = path.split(input_path)[1].split('.')[0]
        fieldnames = fw_results[0].keys()
        with open(path.join(output_dir, input_name + '_fw.csv'), 'w+') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fw_results)

        fieldnames = dw_results[0].keys()
        with open(path.join(output_dir, input_name + '_dw.csv'), 'w+') as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dw_results)

