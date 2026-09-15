"""
Agent A — Researcher

Browses the web (via DuckDuckGo) for the latest news on the given topic,
then asks the LLM to synthesize the raw hits into a clean briefing that
the Writer agent can work from.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from agents.llm import get_llm
from agents.state import AgentState

SYSTEM_PROMPT = """You are Agent A, a meticulous Research Agent on a 3-agent \
editorial team (Researcher -> Writer -> Editor).

You are given a topic and a set of raw web search results (titles, snippets, \
URLs). Your job:
1. Identify the most recent, most relevant, and most credible facts.
2. Discard duplicate or low-value results.
3. Produce a structured research briefing the Writer agent can draft an \
article from directly.

Output format (plain text):
KEY FACTS:
- ...
LATEST DEVELOPMENTS:
- ...
NOTABLE QUOTES / DATA POINTS (if any):
- ...
SOURCES:
- <title> — <url>

Be factual and concise. Do not invent facts that are not supported by the \
search results provided."""


def _web_search(topic: str, max_results: int = 8) -> list[dict]:
    """Run a live DuckDuckGo search. Falls back to a stub if search fails
    (e.g. no internet access in a sandboxed environment) so the pipeline
    can still be demoed end-to-end."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(f"{topic} latest news", max_results=max_results))
        if results:
            return [
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                }
                for r in results
            ]
    except Exception as e:
        print(f"[Researcher] Live web search unavailable ({e}); using offline stub data.")

    # Offline fallback so the graph still runs without internet access.
    return [
        {
            "title": f"[OFFLINE STUB] No live results for '{topic}'",
            "snippet": (
                "Live web search could not be reached from this environment. "
                "Replace this stub by running the project where outbound "
                "internet access to duckduckgo.com is available, or swap in "
                "another search tool (e.g. Tavily, SerpAPI, Bing) in "
                "agents/researcher.py's _web_search()."
            ),
            "url": "",
        }
    ]


def researcher_node(state: AgentState) -> dict:
    topic = state["topic"]
    print(f"\n[Researcher] Searching the web for: {topic!r} ...")

    results = _web_search(topic)

    results_block = "\n\n".join(
        f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}" for r in results
    )

    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Topic: {topic}\n\nRaw search results:\n\n{results_block}\n\n"
            "Produce the research briefing now."
        ),
    ]
    response = llm.invoke(messages)
    notes = response.content

    print("[Researcher] Briefing ready. Handing off to Writer.\n")

    history_entry = {"agent": "Researcher", "action": "gathered research", "output": notes}

    return {
        "search_results": results,
        "research_notes": notes,
        "history": state.get("history", []) + [history_entry],
    }
