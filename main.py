"""
Autonomous AI Agent System — CLI entrypoint.

Usage:
    python main.py "your topic here"
    python main.py                      # prompts for a topic interactively

For the web UI instead, run: python app.py

What happens:
    Agent A (Researcher) searches the web for the latest news on the topic.
    Agent B (Writer) drafts an article from that research.
    Agent C (Editor) reviews the draft; if it needs work, sends feedback
        back to the Writer. This repeats until approved or MAX_REVISIONS
        is hit.
    The final article + a full run log are saved to outputs/.
"""

import sys
from dotenv import load_dotenv
from agents.pipeline import run_pipeline

load_dotenv()

# Prevent UnicodeEncodeError on Windows when the terminal uses cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    topic = " ".join(sys.argv[1:]).strip()
    if not topic:
        topic = input("Enter a topic for the article: ").strip()
    if not topic:
        print("No topic provided. Exiting.")
        return

    print("=" * 70)
    print(" AUTONOMOUS AI AGENT SYSTEM — Researcher -> Writer -> Editor")
    print("=" * 70)
    print(f"Topic: {topic}\n")

    result = run_pipeline(topic)

    print("=" * 70)
    print(f" DONE — approved after {result['revision_count']} revision cycle(s)")
    print(f" Article saved to:  {result['article_path']}")
    print(f" Full run log:      {result['log_path']}")
    print("=" * 70)
    print("\n----- FINAL ARTICLE -----\n")
    print(result["draft"])


if __name__ == "__main__":
    main()
