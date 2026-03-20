from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()


def get_llm(api_key: str):
    """Create LLM using the key passed in — never reads os.environ."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=api_key,
    )


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
    result = (analyzer_prompt | get_llm(state["api_key"]) | parser).invoke({
        "paper_text": state["paper_text"][:40_000]
    })
    return {"analysis": result}


# ── Agent 2: Summary Generator ────────────────
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a scientific writer. Respond with valid JSON only."),
    ("user", """Write an executive summary from this analysis.
Return JSON with keys: executive_summary (150-200 words), word_count.

Analysis: {analysis}""")
])

def summary_generator_agent(state):
    result = (summary_prompt | get_llm(state["api_key"]) | parser).invoke({
        "analysis": state["analysis"]
    })
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
    result = (citation_prompt | get_llm(state["api_key"]) | parser).invoke({
        "paper_text": state["paper_text"][:40_000]
    })
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
    result = (insights_prompt | get_llm(state["api_key"]) | parser).invoke({
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
    review = (review_prompt | get_llm(state["api_key"]) | parser).invoke({
        "task_type":     task_type,
        "content":       content_map.get(task_type, {}),
        "paper_excerpt": state["paper_text"][:2000],
    })
    review["task_type"] = task_type
    return {"review_scores": state.get("review_scores", []) + [review]}


# ── Standalone test ───────────────────────────
if __name__ == "__main__":
    import sys, json, os, PyPDF2
    from dotenv import load_dotenv
    load_dotenv()

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else input("PDF path: ").strip()
    api_key  = os.environ.get("GROQ_API_KEY") or input("Groq API key: ").strip()

    pages = []
    with open(pdf_path, "rb") as f:
        for page in PyPDF2.PdfReader(f).pages:
            if page.extract_text():
                pages.append(page.extract_text())
    paper_text = "\n\n".join(pages)

    print(f"\nLoaded {len(paper_text):,} chars. Running all agents...\n")

    state = {"paper_text": paper_text, "api_key": api_key, "review_scores": []}

    state.update(paper_analyzer_agent(state));  print("✅ Paper Analyzer done")
    state.update(summary_generator_agent(state)); print("✅ Summary Generator done")
    state.update(citation_extractor_agent(state)); print("✅ Citation Extractor done")
    state.update(key_insights_agent(state));     print("✅ Key Insights done")
    state.update(review_agent(state, "analysis")); print("✅ Review Agent done")

    print("\n===== FINAL STATE =====")
    print(json.dumps({
        "analysis":  state["analysis"],
        "summary":   state["summary"],
        "citations": state["citations"],
        "insights":  state["insights"],
        "review":    state["review_scores"],
    }, indent=2))