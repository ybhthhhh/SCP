"""The paper's branch-level and depth-level review-node pruning."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from .graph import Edge, ReasoningGraph


@dataclass(frozen=True)
class PruneResult:
    graph: ReasoningGraph
    removed: frozenset[str]
    branch_redundant: frozenset[str]
    depth_redundant: frozenset[str]


def _descendant_counts(graph: ReasoningGraph) -> dict[str, int]:
    succ = graph.successors()
    counts: dict[str, int] = {}
    for node_id in reversed(graph.topological_order()):
        descendants: set[str] = set()
        for child in succ[node_id]:
            descendants.add(child)
            descendants.update(_collect_descendants(child, succ))
        counts[node_id] = len(descendants)
    return counts


def _collect_descendants(node_id: str, succ: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(succ[node_id])
    while stack:
        item = stack.pop()
        if item in seen:
            continue
        seen.add(item)
        stack.extend(succ[item] - seen)
    return seen


def _shortest_depths(graph: ReasoningGraph) -> dict[str, int]:
    pred = graph.predecessors()
    depth: dict[str, int] = {}
    for node_id in graph.topological_order():
        depth[node_id] = 0 if not pred[node_id] else 1 + min(depth[parent] for parent in pred[node_id])
    return depth


def _contract_removed(graph: ReasoningGraph, removed: set[str]) -> ReasoningGraph:
    """Remove nodes and reconnect kept predecessors to kept successors."""

    pred = graph.predecessors()
    succ = graph.successors()
    kept = {node.id for node in graph.nodes} - removed

    def nearest_kept(start: str, adjacency: dict[str, set[str]]) -> set[str]:
        found: set[str] = set()
        stack = list(adjacency[start])
        seen: set[str] = set()
        while stack:
            item = stack.pop()
            if item in seen:
                continue
            seen.add(item)
            if item in kept:
                found.add(item)
            else:
                stack.extend(adjacency[item])
        return found

    edge_keys: set[tuple[str, str]] = set()
    edges: list[Edge] = []
    for edge in graph.edges:
        if edge.source in kept and edge.target in kept:
            key = (edge.source, edge.target)
            if key not in edge_keys:
                edges.append(edge)
                edge_keys.add(key)
    for removed_id in removed:
        for source in nearest_kept(removed_id, pred):
            for target in nearest_kept(removed_id, succ):
                if source != target and (source, target) not in edge_keys:
                    edges.append(Edge(source=source, target=target, label="via pruned review"))
                    edge_keys.add((source, target))

    return ReasoningGraph(nodes=[node for node in graph.nodes if node.id in kept], edges=edges)


def prune_graph(graph: ReasoningGraph, *, k: int = 2, m: float = 0.9) -> PruneResult:
    """Apply the paper's union of branch and depth redundancy criteria.

    Branch redundancy: review node B(v) < k.
    Depth redundancy: review node d(v) / d_max > m.

    The paper defines d_max using the terminal node. We prefer a node with type
    ``final`` and otherwise use the deepest sink. A zero-depth graph has a ratio
    of zero, so it can only be branch-pruned.
    """

    if k < 0:
        raise ValueError("k must be non-negative")
    if not 0 <= m <= 1:
        raise ValueError("m must be in [0, 1]")

    descendants = _descendant_counts(graph)
    depths = _shortest_depths(graph)
    nodes = graph.node_map()
    finals = [node.id for node in graph.nodes if node.type == "final"]
    if finals:
        terminal = max(finals, key=lambda node_id: depths[node_id])
    else:
        sinks = [node_id for node_id in nodes if not graph.successors()[node_id]]
        terminal = max(sinks, key=lambda node_id: depths[node_id], default=None)
    d_max = depths[terminal] if terminal is not None else 0

    reviews = {node.id for node in graph.nodes if node.type == "review"}
    branch = {node_id for node_id in reviews if descendants[node_id] < k}
    depth = {
        node_id
        for node_id in reviews
        if d_max > 0 and (depths.get(node_id, inf) / d_max) > m
    }
    removed = branch | depth
    return PruneResult(
        graph=_contract_removed(graph, removed),
        removed=frozenset(removed),
        branch_redundant=frozenset(branch),
        depth_redundant=frozenset(depth),
    )


def relinearize(chunks: list[str], graph: ReasoningGraph) -> str:
    """Restore original text order for chunks referenced by retained nodes."""

    indices = sorted({index for node in graph.nodes for index in node.chunks})
    if any(index < 0 or index >= len(chunks) for index in indices):
        raise IndexError("graph contains an out-of-range chunk index")
    return "\n\n".join(chunks[index].strip() for index in indices if chunks[index].strip())
