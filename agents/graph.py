"""
Builds the LangGraph StateGraph connecting the three agents:

    Researcher -> Writer -> Editor -> (approved? END : back to Writer)

The Editor <-> Writer edge is a conditional loop capped by
state["max_revisions"] (see editor.py) so the graph always terminates.
"""

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.researcher import researcher_node
from agents.writer import writer_node
from agents.editor import editor_node


def route_after_editor(state: AgentState) -> str:
    return END if state.get("approved") else "writer"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)

    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "editor")
    graph.add_conditional_edges("editor", route_after_editor, {"writer": "writer", END: END})

    return graph.compile()
