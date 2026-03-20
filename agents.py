"""
agents.py
---------
All agents in one file. Each agent uses LangChain's chain pattern.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

# ── LLM setup ─────────────────────────────────
# Created lazily at call time so the key can come from .env OR user input
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.environ["GROQ_API_KEY"],
    )

parser = JsonOutputParser()


# ── Agent 1: Paper Analyzer ───────────────────
analyzer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research analyst. Respond with valid JSON only."),
    ("user", """Analyze this paper and return JSON with keys:
title, authors (list), year, venue, problem_statement,
hypothesis, methodology, experiments, key_findings (list),
limitations, future_work.

Paper: {paper_text}""")
])

def paper_analyzer_agent(state):
    result = (analyzer_prompt | get_llm() | parser).invoke({"paper_text": state["paper_text"][:40_000]})
    return {"analysis": result}


# ── Agent 2: Summary Generator ────────────────
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a scientific writer. Respond with valid JSON only."),
    ("user", """Write an executive summary from this analysis.
Return JSON with keys: executive_summary (150-200 words), word_count.

Analysis: {analysis}""")
])

def summary_generator_agent(state):
    result = (summary_prompt | get_llm() | parser).invoke({"analysis": state["analysis"]})
    return {"summary": result}


# ── Agent 3: Citation Extractor ───────────────
citation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an academic librarian. Respond with valid JSON only."),
    ("user", """Extract all references from this paper.
Return JSON with keys:
- total_references (integer)
- references (list of: index, authors, title, venue, year)
- key_related_works (list of 3-5 important papers)

Paper: {paper_text}""")
])

def citation_extractor_agent(state):
    result = (citation_prompt | get_llm() | parser).invoke({"paper_text": state["paper_text"][:40_000]})
    return {"citations": result}


# ── Agent 4: Key Insights ─────────────────────
insights_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a tech advisor. Respond with valid JSON only."),
    ("user", """Generate insights from this research.
Return JSON with keys:
- practical_takeaways (list of 3)
- field_implications (string)
- potential_applications (list of 3)
- target_audience (string)
- difficulty_level (Beginner/Intermediate/Advanced)
- recommended_prerequisites (list)

Analysis: {analysis}
Summary: {summary}""")
])

def key_insights_agent(state):
    result = (insights_prompt | get_llm() | parser).invoke({
        "analysis": state["analysis"],
        "summary":  state["summary"],
    })
    return {"insights": result}


# ── Agent 5: Review Agent ─────────────────────
review_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a quality reviewer. Respond with valid JSON only."),
    ("user", """Review this AI output for a research paper task.
Task: {task_type}
Content: {content}
Paper excerpt: {paper_excerpt}

Return JSON with keys:
- score (1-10)
- passed (true if score >= 7)
- strengths (list)
- issues (list)
- improvement_instructions (string, empty if passed)""")
])

def review_agent(state, task_type):
    content_map = {
        "analysis":  state.get("analysis",  {}),
        "summary":   state.get("summary",   {}),
        "citations": state.get("citations", {}),
        "insights":  state.get("insights",  {}),
    }
    review = (review_prompt | get_llm() | parser).invoke({
        "task_type":     task_type,
        "content":       content_map.get(task_type, {}),
        "paper_excerpt": state["paper_text"][:2000],
    })
    review["task_type"] = task_type
    return {"review_scores": state.get("review_scores", []) + [review]}