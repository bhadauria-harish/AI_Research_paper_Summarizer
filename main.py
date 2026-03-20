"""
main.py
-------
CLI entry point for the Research Paper Analyzer.

Usage:
    python main.py --source <path_to_pdf> [--output brief.json]

Example:
    python main.py --source paper.pdf
    python main.py --source paper.pdf --output attention_brief.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from pypdf2 import load_pdf
from workflow import run_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def print_brief(brief: dict) -> None:
    sep = "═" * 60

    meta      = brief.get("paper_metadata", {})
    analysis  = brief.get("research_analysis", {})
    summary   = brief.get("executive_summary", "")
    citations = brief.get("citations_and_references", {})
    insights  = brief.get("key_insights", {})
    scores    = brief.get("quality_scores", {})

    print(f"\n{sep}")
    print("  📄  RESEARCH BRIEF")
    print(sep)
    print(f"\n📌 Title   : {meta.get('title')}")
    print(f"👤 Authors : {', '.join(meta.get('authors', []))}")
    print(f"📅 Year    : {meta.get('year')}   |   🏛  Venue: {meta.get('venue')}")

    print(f"\n{sep}")
    print("  🔬  RESEARCH ANALYSIS")
    print(sep)
    print(f"\n▸ Problem Statement:\n  {analysis.get('problem_statement')}")
    print(f"\n▸ Hypothesis:\n  {analysis.get('hypothesis')}")
    print(f"\n▸ Methodology:\n  {analysis.get('methodology')}")
    print(f"\n▸ Key Findings:")
    for i, f in enumerate(analysis.get("key_findings", []), 1):
        print(f"  {i}. {f}")
    print(f"\n▸ Limitations:\n  {analysis.get('limitations')}")
    print(f"\n▸ Future Work:\n  {analysis.get('future_work')}")

    print(f"\n{sep}")
    print("  📝  EXECUTIVE SUMMARY")
    print(sep)
    print(f"\n{summary}\n")

    print(f"\n{sep}")
    print(f"  📚  CITATIONS  ({citations.get('total_references', 0)} total)")
    print(sep)
    refs = citations.get("references", [])
    for ref in refs[:10]:
        print(f"  [{ref.get('index','?')}] {ref.get('authors','?')} "
              f"({ref.get('year','?')}). {ref.get('title','?')}.")
    if len(refs) > 10:
        print(f"  ... and {len(refs) - 10} more (see JSON output)")
    print(f"\n  Key Related Works:")
    for kw in citations.get("key_related_works", []):
        print(f"  • {kw}")

    print(f"\n{sep}")
    print("  💡  KEY INSIGHTS")
    print(sep)
    print(f"\n▸ Practical Takeaways:")
    for t in insights.get("practical_takeaways", []):
        print(f"  • {t}")
    print(f"\n▸ Field Implications:\n  {insights.get('field_implications')}")

    print(f"\n{sep}")
    print("  ✅  QUALITY SCORES")
    print(sep)
    for task, info in scores.items():
        status = "✅ PASS" if info.get("passed") else "❌ FAIL"
        print(f"  {task:<12} : {info.get('score', '?')}/10  {status}")
    print(f"\n{sep}\n")


def main():
    load_dotenv()

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="AI Research Paper Analyzer")
    parser.add_argument("--source", required=True, help="Path to local PDF file")
    parser.add_argument("--output", default="research_brief.json", help="Output JSON file path")
    args = parser.parse_args()

    # Step 1: Load PDF
    print(f"\n📂 Loading PDF: {args.source}")
    paper_text = load_pdf(args.source)
    print(f"✅ Loaded {len(paper_text):,} characters\n")

    # Step 2: Run workflow
    final_state = run_analysis(paper_text)

    # Step 3: Print & save
    brief = final_state.get("research_brief", {})
    print_brief(brief)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(final_state, indent=2, ensure_ascii=False))
    print(f"💾 Full results saved to: {output_path}")


if __name__ == "__main__":
    main()