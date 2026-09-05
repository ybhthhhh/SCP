"""Graph-construction prompt adapted from Appendix A of the paper."""

GRAPH_UPDATE_SYSTEM = r"""
You analyze and update a chain-of-thought reasoning graph. Given an existing
Mermaid DAG and one current text segment, choose exactly one operation:
Insert or Merge. Return strict JSON only.

A node is one abstract, semantically complete reasoning unit. It should express
an intent and a reusable product. Do not create standalone nodes for low-level
arithmetic, expansion, substitution, or simplification.

Use type "progress" when the unit advances the reasoning frontier. Use type
"review" when it checks, restates, deletes, or rewinds existing reasoning
without advancing the frontier. Review text must never be merged into a
progress node.

Prefer Insert for a new goal, conclusion, method switch, case split, structural
advance, or branch. Use Merge only when the segment continues exactly one node
without mixing progress and review. Add A->B when B uses A's product. New
attempts branch from still-valid ancestors, not dead ends. IDs increase in
reasoning order. The terminal answer node must have id "final answer".

Schema:
{
  "decision": "Insert" or "Merge",
  "target_node": "node id for Merge, otherwise empty",
  "new_node": {
    "id": "new id for Insert, otherwise empty",
    "description": "concise description for Insert, otherwise empty",
    "type": "progress" or "review" or empty for Merge
  },
  "edges": [
    {"from": "source id", "to": "target id", "label": "dependency"}
  ],
  "updated_node_description": "new description for Merge, otherwise empty"
}
""".strip()


def graph_update_user(graph_mermaid: str, current_step: str) -> str:
    return f"Existing Graph:\n{graph_mermaid}\n\nCurrent Step:\n{current_step}\n\nYour Response:"
