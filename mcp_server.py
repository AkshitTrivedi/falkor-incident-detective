"""MCP server: Incident Detective's graph tools for any MCP client.

The same six read-only tools the web agent uses (tools.py), plus the two
actions (actions.py), exposed over the Model Context Protocol. An MCP client
such as Claude Desktop or Claude Code then picks the tools itself; the Cypher
stays fixed and human-written, as in the web app.

The read tools are marked read-only. The two actions are marked as writes, so
clients ask the user before running them, the same "a person approves every
action" rule as the buttons in the UI.

Run (stdio, the transport desktop clients use):
    pip install -r requirements-mcp.txt
    python mcp_server.py

Claude Code:
    claude mcp add incident-detective -- /abs/path/graph-hacks-env/bin/python /abs/path/mcp_server.py

Claude Desktop (claude_desktop_config.json):
    {"mcpServers": {"incident-detective": {
        "command": "/abs/path/graph-hacks-env/bin/python",
        "args": ["/abs/path/mcp_server.py"],
        "env": {"FALKORDB_URL": "falkor://localhost:6379"}}}}

FALKORDB_URL and GRAPH_NAME work as in the web app (see db.py).
"""
import os

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

import actions
import seed
import tools
from db import get_graph

mcp = MCPServer("incident-detective", instructions=(
    "Tools over a FalkorDB graph of microservices, their dependencies, recent deploys, open "
    "alerts, owning teams and on-call people. For 'what broke?': call recent_changes, then "
    "dependencies on the failing service, confirm a suspect deploy with dependency_path, then "
    "who_to_page on the root-cause service. recent_changes also lists incidents already opened, "
    "so check it before paging anyone. open_incident and rollback change the graph: run them "
    "only when the user asks."))

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True,
                        open_world_hint=False)


def _graph():
    """The graph handle, re-seeding the demo story first if it is stale (as app.py does)."""
    g = get_graph()
    if os.getenv("AUTO_RESEED", "1") == "1":
        seed.ensure_fresh(g)
    return g


@mcp.tool(annotations=READ)
def recent_changes(minutes: int = 60) -> dict:
    """Deploys and unresolved alerts from the last `minutes`, newest first, plus incidents
    already opened (who was paged, any rollback). Start here for any 'what broke?' question."""
    return tools.recent_changes(_graph(), minutes)


@mcp.tool(annotations=READ)
def dependencies(service: str) -> dict:
    """Every service `service` calls, directly or indirectly, with hop distance:
    the possible root causes when it fails."""
    return tools.dependencies(_graph(), service)


@mcp.tool(annotations=READ)
def blast_radius(service: str) -> dict:
    """Every service that calls `service`, directly or indirectly: what breaks if it fails.
    An empty list means the graph records no callers, not that nothing can break."""
    return tools.blast_radius(_graph(), service)


@mcp.tool(annotations=READ)
def dependency_path(from_service: str, to_service: str) -> dict:
    """Lowest-latency call path between two services, in either direction (algo.SPpaths).
    path=null means no dependency links them, so an alert on one is unrelated to the other."""
    return tools.dependency_path(_graph(), from_service, to_service)


@mcp.tool(annotations=READ)
def critical_services(metric: str = "betweenness", limit: int = 5) -> dict:
    """Rank services by graph importance: 'betweenness' (chokepoints, for 'most critical')
    or 'pagerank' (most depended on)."""
    return tools.critical_services(_graph(), metric, limit)


@mcp.tool(annotations=READ)
def who_to_page(service: str) -> dict:
    """Owning team and current on-call person for a service (a team name also works)."""
    return tools.who_to_page(_graph(), service)


@mcp.tool(annotations=WRITE)
def open_incident(service: str, version: str) -> dict:
    """Open an incident for the root-cause deploy (`service` `version`) and page the
    on-call person of its owning team. Writes to the graph. Calling it again returns the
    same incident and pages nobody twice."""
    return actions.open_incident(_graph(), service, version)


@mcp.tool(annotations=WRITE)
def rollback(service: str, version: str) -> dict:
    """Roll back the `service` `version` deploy and resolve the alerts on its dependency
    chain; alerts with no path to it stay open. Writes to the graph (demo data only)."""
    return actions.rollback(_graph(), service, version)


if __name__ == "__main__":
    mcp.run()
