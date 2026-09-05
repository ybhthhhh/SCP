"""Typed representation and validation of a reasoning DAG."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


NodeType = Literal["progress", "review", "final"]


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    type: NodeType
    chunks: list[int] = Field(default_factory=list)


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    label: str = "depends on"


class ReasoningGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dag(self) -> "ReasoningGraph":
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        known = set(ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"edge references unknown node: {edge}")
            if edge.source == edge.target:
                raise ValueError("self edges are not allowed")
        self.topological_order()  # raises on cycles
        return self

    def node_map(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def successors(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            result[edge.source].add(edge.target)
        return result

    def predecessors(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            result[edge.target].add(edge.source)
        return result

    def topological_order(self) -> list[str]:
        ids = [node.id for node in self.nodes]
        succ = self.successors()
        indegree = {node_id: 0 for node_id in ids}
        for edge in self.edges:
            indegree[edge.target] += 1
        queue = deque(node_id for node_id in ids if indegree[node_id] == 0)
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for child in succ[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(ids):
            raise ValueError("reasoning graph must be acyclic")
        return order

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for node in self.nodes:
            label = node.description.replace('"', "'")
            lines.append(f'  {node.id}["{label}"]')
        for edge in self.edges:
            label = edge.label.replace('"', "'")
            lines.append(f'  {edge.source} -->|"{label}"| {edge.target}')
        return "\n".join(lines)
