"""
Shared state object that flows between all three agents in the graph.
LangGraph passes this dict-like object node to node, and each node
returns a partial update that gets merged into it.
"""

from typing import TypedDict, List, Optional


class AgentState(TypedDict, total=False):
    topic: str                 # user-provided topic to research/write about
    search_results: List[dict] # raw web search hits gathered by the Researcher
    research_notes: str        # Researcher's synthesized briefing for the Writer
    draft: str                 # current article draft
    feedback: Optional[str]    # Editor's latest revision notes (None once approved)
    approved: bool             # True once the Editor signs off
    revision_count: int        # how many Writer <-> Editor cycles have happened
    max_revisions: int         # hard cap on cycles, from .env
    history: List[dict]        # log of every step, for transparency / debugging
