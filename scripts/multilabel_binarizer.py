import argparse
from os import path
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer

parser = argparse.ArgumentParser(description='Binarize multi-label class on nodes.')
parser.add_argument('input', metavar='input', type=str,
                    help='Path to the .gml file where the graph is stored')

parser.add_argument('output', metavar='output', type=str,
                    help='Path to the dir where to store the resulting .gml file')


def to_list(o):
    if not isinstance(o, list):
        return [o]
    return o


if __name__ == '__main__':
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output

    assert path.exists(input_path)
    assert path.exists(output_dir)
    assert path.isfile(input_path)
    assert path.isdir(output_dir)

    G: nx.Graph = nx.read_gml(input_path)
    binarizer = MultiLabelBinarizer()

    nodes_community = dict(G.nodes(data="community"))
    nodes_community = dict((k, to_list(v)) for k, v in nodes_community.items())

    binarizer.fit(nodes_community.values())
    nodes_community = dict((k, binarizer.transform([v])[0].tolist()) for k, v in nodes_community.items())

    nx.set_node_attributes(G, name="community", values=nodes_community)
    nx.write_gml(G, path.join(output_dir, path.split(input_path)[1].split('.')[0]) + "-bin.gml")
