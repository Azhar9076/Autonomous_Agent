"""
Agent B — Writer

Drafts a news article from the Researcher's briefing. On later calls
(if the Editor requested changes) it revises the previous draft using
the Editor's feedback instead of starting over.
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


def writer_node(state: AgentState) -> dict:
    topic = state["topic"]
    notes = state["research_notes"]
    feedback = state.get("feedback")
    previous_draft = state.get("draft")
    revision_count = state.get("revision_count", 0)

    llm = get_llm(temperature=0.6)

    if feedback and previous_draft:
        print(f"[Writer] Revising draft based on Editor feedback (revision {revision_count}) ...")
        user_content = (
            f"Topic: {topic}\n\nResearch briefing:\n{notes}\n\n"
            f"Previous draft:\n{previous_draft}\n\n"
            f"Editor feedback to address:\n{feedback}\n\n"
            "Write the revised full article now."
        )
    else:
        print("[Writer] Drafting the initial article ...")
        user_content = (
            f"Topic: {topic}\n\nResearch briefing:\n{notes}\n\n"
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
