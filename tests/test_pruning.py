from scp_repro.graph import Edge, Node, ReasoningGraph
from scp_repro.pruning import prune_graph, relinearize


def sample_graph():
    return ReasoningGraph(
        nodes=[
            Node(id="A", description="start", type="progress", chunks=[0]),
            Node(id="B", description="useful review", type="review", chunks=[1]),
            Node(id="C", description="advance", type="progress", chunks=[2]),
            Node(id="D", description="late review", type="review", chunks=[3]),
            Node(id="final answer", description="answer", type="final", chunks=[4]),
        ],
        edges=[
            Edge(source="A", target="B"),
            Edge(source="B", target="C"),
            Edge(source="C", target="D"),
            Edge(source="D", target="final answer"),
        ],
    )


def test_dual_pruning_and_contraction():
    result = prune_graph(sample_graph(), k=2, m=0.7)
    assert result.branch_redundant == {"D"}
    assert result.depth_redundant == {"D"}
    assert result.removed == {"D"}
    assert any(e.source == "C" and e.target == "final answer" for e in result.graph.edges)
    assert relinearize(["a", "b", "c", "d", "answer"], result.graph) == "a\n\nb\n\nc\n\nanswer"


def test_cycle_is_rejected():
    try:
        ReasoningGraph(
            nodes=[Node(id="A", description="a", type="progress"), Node(id="B", description="b", type="progress")],
            edges=[Edge(source="A", target="B"), Edge(source="B", target="A")],
        )
    except ValueError as error:
        assert "acyclic" in str(error)
    else:
        raise AssertionError("cycle was accepted")
