"""
Web UI for the Autonomous AI Agent System.

Run with:  python app.py
Then open: http://localhost:5000
"""

import os
import sys
import glob
import json

from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv

from agents.pipeline import run_pipeline

load_dotenv()

# Prevent UnicodeEncodeError on Windows when the terminal uses cp1252 and
# the LLM outputs characters outside that range (em-dashes, curly quotes, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")

OUTPUT_DIR = "outputs"


def list_runs():
    """Return past runs (newest first) by pairing up each .md with its log."""
    runs = []
    for article_path in sorted(glob.glob(f"{OUTPUT_DIR}/*.md"), reverse=True):
        run_id = os.path.basename(article_path)[:-3]
        log_path = f"{OUTPUT_DIR}/{run_id}-log.json"
        topic, revisions = run_id, None
        if os.path.exists(log_path):
            try:
                with open(log_path, encoding="utf-8") as f:
                    data = json.load(f)
                topic = data.get("topic", run_id)
                revisions = data.get("revision_cycles")
            except Exception:
                pass
        runs.append({"run_id": run_id, "topic": topic, "revisions": revisions})
    return runs


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", runs=list_runs())


@app.route("/run", methods=["POST"])
def run():
    topic = request.form.get("topic", "").strip()
    if not topic:
        flash("Enter a topic before dispatching the newsroom.")
        return redirect(url_for("index"))

    try:
        result = run_pipeline(topic)
    except RuntimeError as e:
        # Missing/invalid API key etc. — surface it plainly instead of a 500.
        flash(str(e))
        return redirect(url_for("index"))

    return redirect(url_for("view_run", run_id=result["run_id"]))


@app.route("/run/<run_id>", methods=["GET"])
def view_run(run_id):
    article_path = f"{OUTPUT_DIR}/{run_id}.md"
    log_path = f"{OUTPUT_DIR}/{run_id}-log.json"

    if not os.path.exists(article_path) or not os.path.exists(log_path):
        flash("That run couldn't be found.")
        return redirect(url_for("index"))

    with open(article_path, encoding="utf-8") as f:
        draft = f.read()
    with open(log_path, encoding="utf-8") as f:
        log_data = json.load(f)

    return render_template(
        "result.html",
        run_id=run_id,
        topic=log_data.get("topic", run_id),
        draft=draft,
        revision_count=log_data.get("revision_cycles", 0),
        research_notes=log_data.get("research_notes", ""),
        history=log_data.get("history", []),
    )


@app.route("/run/<run_id>/delete", methods=["POST"])
def delete_run(run_id):
    safe_run_id = os.path.basename(run_id)
    article_path = f"{OUTPUT_DIR}/{safe_run_id}.md"
    log_path = f"{OUTPUT_DIR}/{safe_run_id}-log.json"

    deleted = False
    if os.path.exists(article_path):
        try:
            os.remove(article_path)
            deleted = True
        except OSError:
            pass

    if os.path.exists(log_path):
        try:
            os.remove(log_path)
            deleted = True
        except OSError:
            pass

    if deleted:
        flash("Edition deleted.")
    else:
        flash("That edition could not be found.")

    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
