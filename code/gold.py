import json
import networkx as nx
from collections import namedtuple


Source = namedtuple("Source", ["book", "author"])


def load_data(path):
    graph = nx.DiGraph()

    with open(path, "r") as f:
        data = json.loads(f.read())

        for k, v in data["nodes"].items():
            v["source"] = Source(v["source"]["book"], v["source"]["author"])
            v["id"] = k
            graph.add_node(k, **v)

        for edge in data["edges"]:
            graph.add_edge(edge[0], edge[1])

    return graph
