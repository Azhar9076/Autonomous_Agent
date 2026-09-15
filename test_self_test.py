"""
Comprehensive self-test suite for Researcher agent and the full pipeline.
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# Force utf-8 encoding for stdout/stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

from agents.researcher import researcher_node
from agents.pipeline import run_pipeline

def test_researcher_queries():
    test_queries = [
        "IndiaAI Mission 2024 artificial intelligence ecosystem",
        "Quantum computing breakthroughs 2024",
        "James Webb Space Telescope recent discoveries",
        "Renewable energy developments in India 2024"
    ]
    
    print("=" * 70)
    print("STAGE 1: TESTING RESEARCHER AGENT (4 QUERIES)")
    print("=" * 70)
    
    all_passed = True
    for idx, query in enumerate(test_queries, 1):
        print(f"\n--- [Test Query {idx}/4]: {query} ---")
        initial_state = {
            "topic": query,
            "history": []
        }
        res = researcher_node(initial_state)
        status = res.get("research_status")
        notes = res.get("research_notes", "")
        results = res.get("search_results", [])
        urls = [r["url"] for r in results if r.get("url")]
        
        print(f"Research Status : {status}")
        print(f"Total Results   : {len(results)}")
        print(f"Unique URLs     : {len(urls)}")
        print("Sample Sources  :")
        for u in urls[:3]:
            print(f"  - {u}")
            
        # Validation checks
        has_min_sources = len(urls) >= 2
        status_ok = status in ("RESEARCH_SUCCESS", "RESEARCH_PARTIAL")
        has_key_facts = "KEY FACTS:" in notes or "KEY FACTS" in notes
        has_developments = "LATEST DEVELOPMENTS:" in notes or "LATEST DEVELOPMENTS" in notes
        no_fake_fail = "no verifiable facts" not in notes.lower() and "no sources retrieved" not in notes.lower()
        
        print(f"Quality Check:")
        print(f"  - Min 2 sources with URL: {'PASS' if has_min_sources else 'FAIL'}")
        print(f"  - Valid status ({status}): {'PASS' if status_ok else 'FAIL'}")
        print(f"  - Key Facts section: {'PASS' if has_key_facts else 'FAIL'}")
        print(f"  - No false-negative text: {'PASS' if no_fake_fail else 'FAIL'}")
        
        if not (has_min_sources and status_ok and no_fake_fail):
            print(f"[!] Test query {idx} failed criteria.")
            all_passed = False
        else:
            print(f"[✓] Test query {idx} PASSED.")
            
    return all_passed

def test_full_pipeline():
    print("\n" + "=" * 70)
    print("STAGE 2: TESTING COMPLETE PIPELINE (RESEARCHER -> WRITER -> EDITOR)")
    print("=" * 70)
    
    test_topic = "Latest developments in India's AI ecosystem and the IndiaAI Mission"
    print(f"Running full pipeline for topic: {test_topic} ...")
    
    result = run_pipeline(test_topic, max_revisions=2)
    
    print("\n--- Pipeline Run Summary ---")
    print(f"Run ID          : {result['run_id']}")
    print(f"Revision Count  : {result['revision_count']}")
    print(f"Article Path    : {result['article_path']}")
    print(f"Log Path        : {result['log_path']}")
    print(f"Draft Length    : {len(result['draft'])} chars")
    
    print("\n--- Final Draft Snippet ---")
    lines = result['draft'].split("\n")
    print("\n".join(lines[:15]))
    print("...")
    
    # Assertions
    assert "Research Failed" not in result['draft'], "Draft incorrectly claims research failed!"
    assert len(result['draft']) > 300, "Draft is too short!"
    assert len(result['history']) >= 3, "Pipeline history incomplete!"
    print("\n[✓] FULL PIPELINE TEST PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    r_ok = test_researcher_queries()
    if r_ok:
        test_full_pipeline()
    else:
        print("\n[X] Researcher tests did not all pass. Review logs above.")
        sys.exit(1)
