"""
Agent C — Editor

Reviews the Writer's draft against the research briefing and either
approves it or sends specific, actionable revision notes back to the
Writer. Responds in a small JSON envelope so the graph can make a
reliable routing decision (approve vs. revise).
"""

import json
import re

from langchain_core.messages import SystemMessage, HumanMessage
from agents.llm import get_llm
from agents.state import AgentState

SYSTEM_PROMPT = """You are Agent C, a strict but fair Editor on a 3-agent \
editorial team (Researcher -> Writer -> Editor).

Review the draft article against the research briefing for:
- Factual accuracy / no unsupported claims
- Clarity and flow
- Structure (headline, dek, inverted pyramid)
- Grammar and tone

Respond with ONLY a JSON object, no other text, no markdown fences:
{
  "approved": true or false,
  "feedback": "If approved: a one-line note saying why it's ready to publish. \
If not approved: a bullet list of specific, actionable revisions the Writer \
must make."
}"""


def _parse_editor_response(raw: str) -> dict:
    """Best-effort JSON extraction in case the model wraps the JSON in
    prose or markdown fences despite instructions."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback: treat as not approved, surface the raw text as feedback.
    return {"approved": False, "feedback": raw.strip()}


def editor_node(state: AgentState) -> dict:
    topic = state["topic"]
    notes = state["research_notes"]
    draft = state["draft"]
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)

    print("[Editor] Reviewing draft ...")

    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Topic: {topic}\n\nResearch briefing:\n{notes}\n\nDraft to review:\n{draft}"
        ),
    ]
    response = llm.invoke(messages)
    parsed = _parse_editor_response(response.content)

    approved = bool(parsed.get("approved", False))
    feedback = parsed.get("feedback", "")

    # Force-approve if we've hit the revision cap, so the graph terminates
    # with the best available draft rather than looping forever.
    hit_cap = revision_count >= max_revisions
    if hit_cap and not approved:
        print(
            f"[Editor] Max revisions ({max_revisions}) reached. "
            "Accepting current draft as final.\n"
        )
        approved = True
        feedback = f"(Auto-approved after {max_revisions} revision cycles) {feedback}"
    elif approved:
        print("[Editor] Draft approved. ✅\n")
    else:
        print(f"[Editor] Requested revisions (cycle {revision_count + 1}):\n{feedback}\n")

    history_entry = {
        "agent": "Editor",
        "action": "approved" if approved else "requested revisions",
        "output": feedback,
    }

    return {
        "approved": approved,
        "feedback": None if approved else feedback,
        "revision_count": revision_count + (0 if approved else 1),
        "history": state.get("history", []) + [history_entry],
    }
