"""FalkorDB connection helper shared by the seed script, the tools and the tests.

Settings come from environment variables:
  FALKORDB_URL  connection URL, e.g. falkor://user:pass@host:port.
                Defaults to the local Docker container.
  GRAPH_NAME    graph (key) to use. Defaults to "incidents".
"""
import os

from falkordb import FalkorDB

GRAPH_NAME = os.environ.get("GRAPH_NAME", "incidents")


def get_graph(name=GRAPH_NAME):
    """Connect to FalkorDB and return a handle to the graph called `name`."""
    # Defaults to local Docker so tests never touch the Cloud instance by accident.
    # The Cloud URL lives in .env as FALKORDB_CLOUD_URL and must be passed in on purpose.
    url = os.environ.get("FALKORDB_URL", "falkor://localhost:6379")
    return FalkorDB.from_url(url).select_graph(name)
