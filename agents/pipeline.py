"""
Runs the full Researcher -> Writer -> Editor graph for a topic and saves
the result to outputs/. Shared by main.py (CLI) and app.py (web UI) so
there's exactly one place that owns this logic.
"""

import os
import json
from datetime import datetime

from agents.graph import build_graph


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")[:40] or "article"


def run_pipeline(topic: str, max_revisions: int = None) -> dict:
    """Run the agent pipeline for `topic`, save the article + log to
    outputs/, and return a dict with everything the UI/CLI needs to
    display the result."""
    if max_revisions is None:
        max_revisions = int(os.getenv("MAX_REVISIONS", "3"))

    graph = build_graph()
    initial_state = {
        "topic": topic,
        "revision_count": 0,
        "max_revisions": max_revisions,
        "approved": False,
        "history": [],
    }

    final_state = graph.invoke(initial_state, config={"recursion_limit": 50})

    os.makedirs("outputs", exist_ok=True)
    slug = slugify(topic)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{slug}-{stamp}"

    article_path = f"outputs/{run_id}.md"
    log_path = f"outputs/{run_id}-log.json"

    with open(article_path, "w", encoding="utf-8") as f:
        f.write(final_state["draft"])

    log_data = {
        "topic": topic,
        "revision_cycles": final_state.get("revision_count", 0),
        "research_notes": final_state.get("research_notes"),
        "history": final_state.get("history", []),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    return {
        "run_id": run_id,
        "topic": topic,
        "draft": final_state["draft"],
        "revision_count": final_state.get("revision_count", 0),
        "research_notes": final_state.get("research_notes", ""),
        "history": final_state.get("history", []),
        "article_path": article_path,
        "log_path": log_path,
    }
