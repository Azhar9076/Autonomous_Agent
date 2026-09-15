# Autonomous AI Agent System — The Wire Desk

A multi-agent pipeline built with **LangGraph** and powered by **Grok (xAI)**,
where three LLM agents collaborate to research and write a news article —
plus a small Flask web UI to run it from your browser instead of the terminal.

| Agent | Role | What it does |
|---|---|---|
| **A — Researcher** | Web research | Searches the web (DuckDuckGo) for the latest news on a topic and synthesizes the results into a structured briefing |
| **B — Writer** | Drafting | Writes an article from the briefing; revises it whenever the Editor sends feedback |
| **C — Editor** | Review | Checks the draft for accuracy, clarity, and structure; approves it or sends specific revision notes back to the Writer |

## Architecture

```
        ┌─────────────┐      ┌───────────┐      ┌───────────┐
 topic─▶│ Researcher  │─────▶│  Writer   │─────▶│  Editor   │
        │  (Agent A)  │      │ (Agent B) │      │ (Agent C) │
        └─────────────┘      └───────────┘      └─────┬─────┘
                                    ▲                   │
                                    │   feedback         │ approved?
                                    └───────────────────┘│
                                      (loop, capped by    │
                                       MAX_REVISIONS)     ▼
                                                         END
                                                  (article saved)
```

Implemented as a `langgraph.graph.StateGraph` (`agents/graph.py`):
- `researcher -> writer` (always)
- `writer -> editor` (always)
- `editor -> writer` **if not approved**, `editor -> END` **if approved**

A shared `AgentState` (`agents/state.py`) flows between every node. The
full history of every agent's output is kept in `state["history"]` and
shown in both the CLI output and the web UI's "newsroom trace".

The Editor↔Writer loop is capped by `MAX_REVISIONS` (default 3, in `.env`)
so the graph always terminates — after the cap, the Editor auto-approves
the best available draft.

## LLM: Groq by default

`agents/llm.py` is the single place that builds the chat model. It
defaults to **Groq** (`openai/gpt-oss-120b`) via Groq's OpenAI-compatible
API (`https://api.groq.com/openai/v1`), and can be switched to Grok
(xAI), Anthropic, or OpenAI by changing `LLM_PROVIDER` in `.env` — no
code changes needed.

> **Groq vs. Grok:** these are two different companies with similar
> names. **Groq** (console.groq.com, keys start with `gsk_`) makes fast
> inference hardware and serves open models like Llama and GPT-OSS.
> **Grok** (console.x.ai, keys start with `xai-`) is xAI's own model.
> Getting these mixed up is the #1 source of "incorrect API key" errors
> with this project — check `.env.example` for both options.

## Project structure

```
autonomous-agent-system/
├── app.py                    # Flask web UI
├── main.py                   # CLI entrypoint
├── templates/
│   ├── index.html             # dispatch form + archive of past runs
│   └── result.html            # article + newsroom trace
├── static/
│   └── style.css              # wire-service / newsroom theme
├── agents/
│   ├── state.py                # shared AgentState TypedDict
│   ├── llm.py                   # LLM factory (Grok / Anthropic / OpenAI)
│   ├── pipeline.py             # shared run_pipeline() used by CLI + UI
│   ├── researcher.py           # Agent A
│   ├── writer.py                # Agent B
│   ├── editor.py                # Agent C
│   └── graph.py                  # LangGraph wiring
├── outputs/                   # generated articles + run logs land here
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your XAI_API_KEY (get one at https://console.x.ai)
```

## Run — web UI

```bash
python app.py
```

Open **http://localhost:5000**. Type a topic into "Assign a topic" and hit
**Dispatch**. The request runs synchronously (usually a minute or two —
Researcher, then Writer, then possibly a couple of Editor↔Writer revision
rounds), then you land on a result page with:
- the final article, styled as a newspaper column
- a **newsroom trace** — every agent's output in order, click to expand
- an archive of past runs on the home page

## Run — CLI

```bash
python main.py "ISRO Gaganyaan mission update"
```

Both the CLI and the web UI call the same `agents/pipeline.py`, so results
are identical either way — each run produces:
- `outputs/<topic>-<timestamp>.md` — the final, approved article
- `outputs/<topic>-<timestamp>-log.json` — full trace of every agent's
  output and how many revision cycles it took

## Notes for viva / demo

- **Why LangGraph over plain function calls?** LangGraph models the
  workflow explicitly as a graph with state, so the Editor→Writer
  revision loop, the termination condition, and the full audit trail
  come for free instead of being hand-rolled with `while` loops.
- **Why a JSON envelope for the Editor?** The Editor's output has to
  drive a routing decision (loop vs. stop), not just be read by a
  human, so it responds with `{"approved": bool, "feedback": str}`
  instead of free text.
- **Why is Grok called through `ChatOpenAI`?** xAI's API is
  OpenAI-compatible, so `langchain_openai.ChatOpenAI` pointed at
  `base_url="https://api.x.ai/v1"` with an `XAI_API_KEY` works without
  needing a separate Grok-specific client library.
- **Offline fallback:** if the environment running this has no
  outbound internet access, `agents/researcher.py` falls back to a
  labeled stub result so the rest of the pipeline can still be
  demonstrated end-to-end.
- **Swapping providers:** change `LLM_PROVIDER` in `.env` between
  `grok`, `anthropic`, and `openai` — no code changes.
