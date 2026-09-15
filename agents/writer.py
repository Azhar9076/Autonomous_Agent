"""
Agent B — Writer

Drafts a news article from the Researcher's briefing. On later calls
(if the Editor requested changes) it revises the previous draft using
the Editor's feedback instead of starting over.

Guard: if research_status is SEARCH_FAILED, the Writer refuses to fabricate
an article and returns a research-failure notice instead.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.llm import get_llm
from agents.state import AgentState

SYSTEM_PROMPT = """You are Agent B, a skilled Writer on a 3-agent editorial \
team (Researcher -> Writer -> Editor).

You write clear, engaging, factually grounded news articles from a research \
briefing. Structure:
- A strong headline
- A one-line dek/subheading
- 4-7 short paragraphs in inverted-pyramid style (most important info first)
- Neutral, journalistic tone

When you receive Editor feedback on a previous draft, revise that exact \
draft to address every point raised — do not ignore feedback, and do not \
introduce facts that aren't in the research briefing."""

_SEARCH_FAILED_DRAFT = """\
# Research Failed — Article Not Generated

**Topic:** {topic}

The Researcher agent was unable to retrieve any web search results for this \
topic. No article has been written because there is no verified evidence to \
work from.

**What this means:**
- The web search service may be temporarily unavailable or rate-limiting requests.
- Try again in a few minutes.
- If the problem persists, check that the server has outbound internet access.

**Research status:** SEARCH_FAILED
"""


def writer_node(state: AgentState) -> dict:
    topic = state["topic"]
    research_status = state.get("research_status", "RESEARCH_SUCCESS")
    notes = state.get("research_notes", "")
    feedback = state.get("feedback")
    previous_draft = state.get("draft")
    revision_count = state.get("revision_count", 0)

    # ── Guard: do NOT write an article if search failed ──────────────────────
    if research_status == "SEARCH_FAILED":
        print("[Writer] ✗ research_status=SEARCH_FAILED — refusing to fabricate article.")
        draft = _SEARCH_FAILED_DRAFT.format(topic=topic)
        history_entry = {
            "agent": "Writer",
            "action": "refused to write (SEARCH_FAILED)",
            "output": draft,
        }
        return {
            "draft": draft,
            "approved": True,   # force graph to terminate — no point in Editor loop
            "history": state.get("history", []) + [history_entry],
        }

    # ── Partial / no-evidence warning injected into prompt ───────────────────
    if research_status in ("RESEARCH_PARTIAL", "NO_EVIDENCE_AFTER_SEARCH"):
        print(f"[Writer] ⚠ research_status={research_status} — adding caution to prompt.")
        extra_caution = (
            "\n\nIMPORTANT: The research briefing is incomplete or of limited quality. "
            "Write ONLY what is supported by the sources listed. If a claim has no source, "
            "omit it. Do NOT invent facts."
        )
    else:
        extra_caution = ""

    llm = get_llm(temperature=0.6)

    if feedback and previous_draft:
        print(f"[Writer] Revising draft based on Editor feedback (revision {revision_count}) ...")
        user_content = (
            f"Topic: {topic}\n\nResearch briefing:\n{notes}{extra_caution}\n\n"
            f"Previous draft:\n{previous_draft}\n\n"
            f"Editor feedback to address:\n{feedback}\n\n"
            "Write the revised full article now."
        )
    else:
        print("[Writer] Drafting the initial article ...")
        user_content = (
            f"Topic: {topic}\n\nResearch briefing:\n{notes}{extra_caution}\n\n"
            "Write the initial full article now."
        )

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)]
    response = llm.invoke(messages)
    draft = response.content

    print("[Writer] Draft ready. Sending to Editor.\n")

    action = "revised draft" if feedback else "wrote initial draft"
    history_entry = {"agent": "Writer", "action": action, "output": draft}

    return {
        "draft": draft,
        "history": state.get("history", []) + [history_entry],
    }
