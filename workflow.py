import os
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langsmith import traceable
from dotenv import load_dotenv

from agents import (
    paper_analyzer_agent,
    summary_generator_agent,
    citation_extractor_agent,
    key_insights_agent,
    review_agent,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

MAX_RETRIES = 2


# ── State ──────────────────────────────────────
def merge_lists(a, b):
    return (a or []) + (b or [])

class ResearchState(TypedDict):
    api_key:        str
    paper_text:     str
    analysis:       dict
    summary:        dict
    citations:      dict
    insights:       dict
    review_scores:  Annotated[list, merge_lists]
    retry_counts:   dict
    research_brief: dict


# ── Nodes ──────────────────────────────────────
def node_paper_analyzer(state):
    return paper_analyzer_agent(state)

def node_review_analysis(state):
    return review_agent(state, "analysis")

def node_summary_generator(state):
    return summary_generator_agent(state)

def node_review_summary(state):
    return review_agent(state, "summary")

def node_citation_extractor(state):
    return citation_extractor_agent(state)

def node_review_citations(state):
    return review_agent(state, "citations")

def node_key_insights(state):
    return key_insights_agent(state)

def node_review_insights(state):
    return review_agent(state, "insights")

def node_combiner(state):
    """Boss agent — merges all outputs into the final research brief."""
    analysis  = state.get("analysis",  {})
    summary   = state.get("summary",   {})
    citations = state.get("citations", {})
    insights  = state.get("insights",  {})

    # Build score summary from review history
    scores = {
        r["task_type"]: {"score": r.get("score"), "passed": r.get("passed")}
        for r in state.get("review_scores", [])
        if "task_type" in r
    }

    brief = {
        "paper_metadata": {
            "title":   analysis.get("title",   "Unknown"),
            "authors": analysis.get("authors", []),
            "year":    analysis.get("year",    "Unknown"),
            "venue":   analysis.get("venue",   "Unknown"),
        },
        "research_analysis": {
            "problem_statement": analysis.get("problem_statement", ""),
            "hypothesis":        analysis.get("hypothesis",        ""),
            "methodology":       analysis.get("methodology",       ""),
            "experiments":       analysis.get("experiments",       ""),
            "key_findings":      analysis.get("key_findings",      []),
            "limitations":       analysis.get("limitations",       ""),
            "future_work":       analysis.get("future_work",       ""),
        },
        "executive_summary": summary.get("executive_summary", ""),
        "citations_and_references": {
            "total_references":  citations.get("total_references", 0),
            "references":        citations.get("references",       []),
            "key_related_works": citations.get("key_related_works",[]),
        },
        "key_insights": {
            "practical_takeaways":       insights.get("practical_takeaways",       []),
            "field_implications":        insights.get("field_implications",        ""),
            "potential_applications":    insights.get("potential_applications",    []),
            "target_audience":           insights.get("target_audience",           ""),
            "difficulty_level":          insights.get("difficulty_level",          ""),
            "recommended_prerequisites": insights.get("recommended_prerequisites", []),
        },
        "quality_scores": scores,
    }
    return {"research_brief": brief}


# ── Retry helpers ──────────────────────────────
def _get_latest_review(state, task_type):
    reviews = [r for r in state.get("review_scores", []) if r.get("task_type") == task_type]
    return reviews[-1] if reviews else None

def _get_retry_count(state, task):
    return state.get("retry_counts", {}).get(task, 0)

def _bump_retry(state, task):
    counts = dict(state.get("retry_counts", {}))
    counts[task] = counts.get(task, 0) + 1
    return {"retry_counts": counts}


# ── Routing functions ──────────────────────────
def route_analysis(state):
    review = _get_latest_review(state, "analysis")
    if (review and review.get("passed")) or _get_retry_count(state, "analysis") >= MAX_RETRIES:
        return "go_summary"
    return "retry_analysis"

def route_summary(state):
    review = _get_latest_review(state, "summary")
    if (review and review.get("passed")) or _get_retry_count(state, "summary") >= MAX_RETRIES:
        return "go_citations"
    return "retry_summary"

def route_citations(state):
    review = _get_latest_review(state, "citations")
    if (review and review.get("passed")) or _get_retry_count(state, "citations") >= MAX_RETRIES:
        return "go_insights"
    return "retry_citations"

def route_insights(state):
    review = _get_latest_review(state, "insights")
    if (review and review.get("passed")) or _get_retry_count(state, "insights") >= MAX_RETRIES:
        return "go_combine"
    return "retry_insights"


# ── Retry nodes ────────────────────────────────
def node_retry_analysis(state):
    return {**paper_analyzer_agent(state), **_bump_retry(state, "analysis")}

def node_retry_summary(state):
    return {**summary_generator_agent(state), **_bump_retry(state, "summary")}

def node_retry_citations(state):
    return {**citation_extractor_agent(state), **_bump_retry(state, "citations")}

def node_retry_insights(state):
    return {**key_insights_agent(state), **_bump_retry(state, "insights")}


# ── Build graph ────────────────────────────────
def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("paper_analyzer",     node_paper_analyzer)
    g.add_node("review_analysis",    node_review_analysis)
    g.add_node("retry_analysis",     node_retry_analysis)

    g.add_node("summary_generator",  node_summary_generator)
    g.add_node("review_summary",     node_review_summary)
    g.add_node("retry_summary",      node_retry_summary)

    g.add_node("citation_extractor", node_citation_extractor)
    g.add_node("review_citations",   node_review_citations)
    g.add_node("retry_citations",    node_retry_citations)

    g.add_node("key_insights",       node_key_insights)
    g.add_node("review_insights",    node_review_insights)
    g.add_node("retry_insights",     node_retry_insights)

    g.add_node("combiner",           node_combiner)

    g.set_entry_point("paper_analyzer")

    g.add_edge("paper_analyzer", "review_analysis")
    g.add_conditional_edges("review_analysis", route_analysis,
        {"go_summary": "summary_generator", "retry_analysis": "retry_analysis"})
    g.add_edge("retry_analysis", "review_analysis")

    g.add_edge("summary_generator", "review_summary")
    g.add_conditional_edges("review_summary", route_summary,
        {"go_citations": "citation_extractor", "retry_summary": "retry_summary"})
    g.add_edge("retry_summary", "review_summary")

    g.add_edge("citation_extractor", "review_citations")
    g.add_conditional_edges("review_citations", route_citations,
        {"go_insights": "key_insights", "retry_citations": "retry_citations"})
    g.add_edge("retry_citations", "review_citations")

    g.add_edge("key_insights", "review_insights")
    g.add_conditional_edges("review_insights", route_insights,
        {"go_combine": "combiner", "retry_insights": "retry_insights"})
    g.add_edge("retry_insights", "review_insights")

    g.add_edge("combiner", END)

    return g.compile()


# ── Run function ───────────────────────────────
@traceable(name="research-paper-analyzer")
def run_analysis(paper_text: str, api_key: str) -> dict:
    app = build_graph()
    return app.invoke({
        "api_key":        api_key,
        "paper_text":     paper_text,
        "analysis":       {},
        "summary":        {},
        "citations":      {},
        "insights":       {},
        "review_scores":  [],
        "retry_counts":   {},
        "research_brief": {},
    })