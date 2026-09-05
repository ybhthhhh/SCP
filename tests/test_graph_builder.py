import pytest

from scp_repro.builder import GraphBuilder, GraphUpdate, parse_json_object
from scp_repro.graph import ReasoningGraph


def test_insert_then_merge():
    graph = ReasoningGraph()
    insert = GraphUpdate.model_validate(
        {
            "decision": "Insert",
            "target_node": "",
            "new_node": {"id": "A", "description": "Set a goal", "type": "progress"},
            "edges": [],
            "updated_node_description": "",
        }
    )
    graph = GraphBuilder.apply(graph, insert, chunk_index=0)
    merge = GraphUpdate.model_validate(
        {
            "decision": "Merge",
            "target_node": "A",
            "new_node": {"id": "", "description": "", "type": ""},
            "edges": [],
            "updated_node_description": "Set and solve the goal",
        }
    )
    graph = GraphBuilder.apply(graph, merge, chunk_index=1)
    assert graph.nodes[0].chunks == [0, 1]
    assert graph.nodes[0].description == "Set and solve the goal"


def test_parse_fenced_json():
    assert parse_json_object('```json\n{"decision": "Insert"}\n```')["decision"] == "Insert"


def test_unknown_edge_rejected():
    update = GraphUpdate.model_validate(
        {
            "decision": "Insert",
            "new_node": {"id": "A", "description": "x", "type": "progress"},
            "edges": [{"from": "Z", "to": "A", "label": "bad"}],
        }
    )
    with pytest.raises(ValueError, match="unknown"):
        GraphBuilder.apply(ReasoningGraph(), update, chunk_index=0)
