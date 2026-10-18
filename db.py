"""FalkorDB connection helper shared by the seed script, the tools and the tests.

Settings come from environment variables:
  FALKORDB_URL  connection URL, e.g. falkor://user:pass@host:port.
                Defaults to the local Docker container.
  GRAPH_NAME    graph (key) to use. Defaults to "incidents".
"""
import os
from functools import lru_cache

from falkordb import FalkorDB

GRAPH_NAME = os.environ.get("GRAPH_NAME", "incidents")


@lru_cache(maxsize=None)
def _client(url):
    """One FalkorDB client per URL, shared by every request.

    Opening a connection to FalkorDB Cloud costs 1-4 s from the app host, so
    reconnecting per request made a 'hello' take 12 s. The client's pool is
    thread-safe and reconnects by itself if the link drops.
    """
    return FalkorDB.from_url(url)


def get_graph(name=GRAPH_NAME):
    """Connect to FalkorDB and return a handle to one graph.

    Selecting a graph that does not exist yet is fine; FalkorDB creates it on
    the first write (seed.py does that).

    Args:
        name: Graph name. Defaults to GRAPH_NAME.

    Returns:
        falkordb.Graph, used for .query() (writes, seed only) and .ro_query().
    """
    # Defaults to local Docker so tests never touch the Cloud instance by accident.
    # The Cloud URL lives in .env as FALKORDB_CLOUD_URL and must be passed in on purpose.
    url = os.environ.get("FALKORDB_URL", "falkor://localhost:6379")
    return _client(url).select_graph(name)
