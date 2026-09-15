"""
Agent A — Researcher

Searches the web (via the `ddgs` library — successor to duckduckgo-search)
for the latest information on the given topic, then asks the LLM to synthesise
the raw hits into a clean briefing that the Writer agent can work from.

Key improvements over the original:
  - Uses `ddgs` instead of the deprecated `duckduckgo_search` package.
  - Runs MULTIPLE search queries (varied phrasing) for robustness.
  - Retries each query up to MAX_RETRIES times if it returns 0 results.
  - Deduplicates results by URL across all queries.
  - Applies a quality gate: at least MIN_SOURCES results with a real URL
    must be found before calling the LLM.
  - Sets state["research_status"] so downstream agents (Writer / Editor)
    can react correctly without guessing from content strings.
  - Never silently swallows exceptions — all failures are printed and
    escalated to an explicit SEARCH_FAILED status.
"""

import time
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from agents.llm import get_llm
from agents.state import AgentState

logger = logging.getLogger(__name__)

# ─── tunables ────────────────────────────────────────────────────────────────
MAX_RETRIES = 3          # per-query retry attempts
RETRY_DELAY = 2.0        # seconds to wait between retries
MAX_RESULTS_PER_QUERY = 8
MIN_SOURCES = 2          # minimum real-URL results required for RESEARCH_SUCCESS
# ─────────────────────────────────────────────────────────────────────────────

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

IMPORTANT RULES:
- Every point under KEY FACTS and LATEST DEVELOPMENTS MUST be traceable to
  one of the provided search results.
- List at least as many SOURCES entries as there are search results with
  non-empty URLs.
- Do NOT invent facts that are not in the provided search results.
- If the search results do not contain information about the topic, say so
  explicitly — do NOT fabricate information."""


def _build_queries(topic: str) -> list[str]:
    """Generate multiple search query variants for better coverage."""
    return [
        f"{topic} latest news 2024 2025",
        f"{topic} recent developments",
        f"{topic} overview facts",
        f"{topic}",
    ]


def _search_once(query: str) -> list[dict]:
    """
    Attempt a single DuckDuckGo text search using the `ddgs` library.
    Returns a list of result dicts (title, snippet, url).
    Raises on failure so the caller can retry.
    """
    from ddgs import DDGS

    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=MAX_RESULTS_PER_QUERY))

    return [
        {
            "title": r.get("title", "").strip(),
            "snippet": r.get("body", "").strip(),
            "url": r.get("href", "").strip(),
        }
        for r in raw
        if r.get("title") or r.get("body")  # keep results that have at least something
    ]


def _web_search(topic: str) -> tuple[list[dict], str]:
    """
    Run multiple query variants, retry on failure/empty, deduplicate by URL.

    Returns:
        (results, status) where status is one of the ResearchStatus literals.
    """
    queries = _build_queries(topic)
    seen_urls: set[str] = set()
    all_results: list[dict] = []
    any_exception = False

    for query in queries:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[Researcher] Query (attempt {attempt}/{MAX_RETRIES}): {query!r}")
                hits = _search_once(query)
                print(f"[Researcher]   → {len(hits)} results returned")

                if hits:
                    for h in hits:
                        url = h.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(h)
                        elif not url:
                            # Keep results without URL too (snippet may still be useful)
                            all_results.append(h)
                    break  # success for this query — move to next query
                else:
                    print(f"[Researcher]   → empty result set, retrying in {RETRY_DELAY}s …")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)

            except Exception as exc:
                any_exception = True
                logger.warning("[Researcher] Search error on attempt %d for %r: %s: %s",
                               attempt, query, type(exc).__name__, exc)
                print(f"[Researcher] Search error (attempt {attempt}): {type(exc).__name__}: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

    # Count results that have a real URL (quality metric)
    real_results = [r for r in all_results if r.get("url")]
    print(f"[Researcher] Total unique results: {len(all_results)} "
          f"({len(real_results)} with URLs)")

    if not all_results:
        status = "SEARCH_FAILED"
    elif len(real_results) >= MIN_SOURCES:
        status = "RESEARCH_SUCCESS"
    else:
        status = "RESEARCH_PARTIAL"

    return all_results, status


def researcher_node(state: AgentState) -> dict:
    topic = state["topic"]
    print(f"\n[Researcher] ── Starting research for: {topic!r} ──")

    results, search_status = _web_search(topic)

    if search_status == "SEARCH_FAILED":
        print("[Researcher] ✗ All search attempts failed. Flagging SEARCH_FAILED.")
        notes = (
            f"RESEARCH STATUS: SEARCH_FAILED\n\n"
            f"The web search could not retrieve any results for the topic: {topic!r}.\n"
            f"This may be due to network restrictions, rate-limiting, or the search "
            f"service being unavailable.\n\n"
            f"The Writer MUST NOT produce an article — there is no evidence to work from."
        )
        history_entry = {
            "agent": "Researcher",
            "action": "search failed",
            "output": notes,
        }
        return {
            "search_results": [],
            "research_notes": notes,
            "research_status": "SEARCH_FAILED",
            "history": state.get("history", []) + [history_entry],
        }

    # Build the results block for the LLM
    results_block = "\n\n".join(
        f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}"
        for r in results
    )

    print(f"[Researcher] Sending {len(results)} results to LLM for synthesis …")
    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Topic: {topic}\n\n"
                f"Raw search results:\n\n{results_block}\n\n"
                "Produce the research briefing now."
            )
        ),
    ]
    response = llm.invoke(messages)
    notes = response.content.strip()

    # Sanity-check: did the LLM actually produce evidence?
    failure_phrases = [
        "no verifiable facts",
        "no verifiable developments",
        "no sources retrieved",
        "no live results",
        "offline stub",
        "information is not available",
    ]
    notes_lower = notes.lower()
    llm_empty = any(phrase in notes_lower for phrase in failure_phrases)

    if llm_empty and search_status == "RESEARCH_SUCCESS":
        # We had good sources but LLM still produced empty output — escalate
        print("[Researcher] ⚠ LLM returned empty briefing despite valid search results. "
              "Flagging NO_EVIDENCE_AFTER_SEARCH.")
        research_status = "NO_EVIDENCE_AFTER_SEARCH"
    elif search_status == "RESEARCH_PARTIAL":
        research_status = "RESEARCH_PARTIAL"
    else:
        research_status = "RESEARCH_SUCCESS"

    print(f"[Researcher] ✓ Research complete — status: {research_status}")
    print("[Researcher] Briefing ready. Handing off to Writer.\n")

    history_entry = {
        "agent": "Researcher",
        "action": f"gathered research [{research_status}]",
        "output": notes,
    }

    return {
        "search_results": results,
        "research_notes": notes,
        "research_status": research_status,
        "history": state.get("history", []) + [history_entry],
    }
