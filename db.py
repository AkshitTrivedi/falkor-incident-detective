import os

from falkordb import FalkorDB

GRAPH_NAME = os.environ.get("GRAPH_NAME", "incidents")


def get_graph(name=GRAPH_NAME):
    # Defaults to local Docker so tests never touch the Cloud instance by accident.
    url = os.environ.get("FALKORDB_URL", "falkor://localhost:6379")
    return FalkorDB.from_url(url).select_graph(name)
