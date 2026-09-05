"""Iterative LLM graph construction with strict update validation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .graph import Edge, Node, ReasoningGraph
from .prompts import GRAPH_UPDATE_SYSTEM, graph_update_user


class UpdateNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""
    description: str = ""
    type: Literal["", "progress", "review"] = ""


class UpdateEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    source: str = Field(alias="from")
    target: str = Field(alias="to")
    label: str = "depends on"


class GraphUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["Insert", "Merge"]
    target_node: str = ""
    new_node: UpdateNode = Field(default_factory=UpdateNode)
    edges: list[UpdateEdge] = Field(default_factory=list)
    updated_node_description: str = ""

    @model_validator(mode="after")
    def validate_operation(self) -> "GraphUpdate":
        if self.decision == "Insert" and not (self.new_node.id and self.new_node.description):
            raise ValueError("Insert requires a new node id and description")
        if self.decision == "Merge" and not (self.target_node and self.updated_node_description):
            raise ValueError("Merge requires target_node and updated_node_description")
        return self


def parse_json_object(text: str) -> dict:
    """Parse plain or fenced model JSON without accepting trailing prose."""

    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("graph update must be a JSON object")
    return value


class GraphBuilder:
    """Apply one model-produced operation per original CoT chunk."""

    def __init__(self, complete: Callable[[str, str], str]):
        self.complete = complete

    def build(self, chunks: list[str]) -> ReasoningGraph:
        graph = ReasoningGraph()
        for index, chunk in enumerate(chunks):
            raw = self.complete(GRAPH_UPDATE_SYSTEM, graph_update_user(graph.to_mermaid(), chunk))
            update = GraphUpdate.model_validate(parse_json_object(raw))
            graph = self.apply(graph, update, chunk_index=index)
        return graph

    @staticmethod
    def apply(graph: ReasoningGraph, update: GraphUpdate, *, chunk_index: int) -> ReasoningGraph:
        nodes = [node.model_copy(deep=True) for node in graph.nodes]
        edges = [edge.model_copy(deep=True) for edge in graph.edges]
        node_map = {node.id: node for node in nodes}

        if update.decision == "Insert":
            node_id = update.new_node.id
            if node_id in node_map:
                raise ValueError(f"Insert reused node id {node_id!r}")
            if update.new_node.type not in {"progress", "review"}:
                raise ValueError("Insert requires progress or review type")
            node_type = "final" if node_id.strip().lower() == "final answer" else update.new_node.type
            nodes.append(
                Node(
                    id=node_id,
                    description=update.new_node.description,
                    type=node_type,
                    chunks=[chunk_index],
                )
            )
        else:
            if update.target_node not in node_map:
                raise ValueError(f"Merge target does not exist: {update.target_node!r}")
            target = node_map[update.target_node]
            if target.type == "progress" and update.new_node.type == "review":
                raise ValueError("review content cannot be merged into a progress node")
            target.description = update.updated_node_description
            target.chunks.append(chunk_index)

        known = {node.id for node in nodes}
        existing = {(edge.source, edge.target) for edge in edges}
        for item in update.edges:
            if item.source not in known or item.target not in known:
                raise ValueError(f"new edge references an unknown node: {item.source}->{item.target}")
            if (item.source, item.target) not in existing:
                edges.append(Edge(source=item.source, target=item.target, label=item.label))
                existing.add((item.source, item.target))
        return ReasoningGraph(nodes=nodes, edges=edges)
