import json
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from pypdf2 import load_pdf
from workflow import run_analysis

load_dotenv()

st.set_page_config(page_title="Research Paper Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI Research Paper Analyzer")
st.caption("Multi-agent system · LangGraph + Groq (llama-3.3-70b)")

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.divider()
    st.markdown("""
**Agent Pipeline:**
1. 📊 Paper Analyzer
2. 📝 Summary Generator
3. 📚 Citation Extractor
4. 💡 Key Insights
5. ✅ Review Agent *(scores each output)*
6. 🎯 Boss Combiner
""")

# ── Upload ─────────────────────────────────────
st.subheader("📥 Upload Paper")
uploaded_file = st.file_uploader("Upload a research paper PDF", type=["pdf"])
run_btn = st.button("🚀 Analyze Paper", type="primary", use_container_width=True)

# ── Run ────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()
    if not uploaded_file:
        st.error("Please upload a PDF first.")
        st.stop()

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    paper_text = load_pdf(tmp_path)
    st.success(f"✅ Loaded {len(paper_text):,} characters")

    with st.spinner("Running agents... this may take a minute ⏳"):
        try:
            final_state = run_analysis(paper_text)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    st.session_state["brief"]      = final_state.get("research_brief", {})
    st.session_state["full_state"] = final_state
    st.success("✅ Analysis complete!")

# ── Display Results ────────────────────────────
if "brief" in st.session_state:
    brief      = st.session_state["brief"]
    full_state = st.session_state["full_state"]

    st.divider()
    meta = brief.get("paper_metadata", {})

    # Header
    st.markdown(f"## {meta.get('title', 'Unknown Title')}")
    st.caption(f"**Authors:** {', '.join(meta.get('authors', []))} | **Year:** {meta.get('year')} | **Venue:** {meta.get('venue')}")

    col1, col2 = st.columns(2)
    col1.metric("Total References", brief.get("citations_and_references", {}).get("total_references", 0))
    col2.metric("Difficulty", brief.get("key_insights", {}).get("difficulty_level", "N/A"))

    # Tabs
    t1, t2, t3, t4, t5 = st.tabs(["🔬 Analysis", "📝 Summary", "📚 Citations", "💡 Insights", "📊 Scores"])

    with t1:
        a = brief.get("research_analysis", {})
        st.info(f"**Problem:** {a.get('problem_statement','')}")
        st.write(f"**Hypothesis:** {a.get('hypothesis','')}")
        st.write(f"**Methodology:** {a.get('methodology','')}")
        st.write("**Key Findings:**")
        for f in a.get("key_findings", []):
            st.markdown(f"- {f}")
        with st.expander("More details"):
            st.write(f"**Experiments:** {a.get('experiments','')}")
            st.write(f"**Limitations:** {a.get('limitations','')}")
            st.write(f"**Future Work:** {a.get('future_work','')}")

    with t2:
        st.write(brief.get("executive_summary", ""))

    with t3:
        c = brief.get("citations_and_references", {})
        st.write("**Key Related Works:**")
        for kw in c.get("key_related_works", []):
            st.markdown(f"- {kw}")
        st.write("**All References:**")
        for ref in c.get("references", []):
            st.markdown(f"[{ref.get('index','?')}] **{ref.get('title','?')}** — {ref.get('authors','?')} ({ref.get('year','?')})")

    with t4:
        i = brief.get("key_insights", {})
        st.write(f"**Target Audience:** {i.get('target_audience','')}")
        st.write(f"**Field Implications:** {i.get('field_implications','')}")
        st.write("**Practical Takeaways:**")
        for t in i.get("practical_takeaways", []):
            st.markdown(f"✅ {t}")
        st.write("**Potential Applications:**")
        for app in i.get("potential_applications", []):
            st.markdown(f"🚀 {app}")

    with t5:
        for task, info in brief.get("quality_scores", {}).items():
            passed = info.get("passed", False)
            st.markdown(f"**{task}** — {info.get('score')}/10 {'✅' if passed else '❌'}")
        st.divider()
        st.write("**Review History:**")
        for review in full_state.get("review_scores", []):
            with st.expander(f"{review.get('task_type')} — {review.get('score')}/10"):
                st.write(f"Strengths: {', '.join(review.get('strengths', []))}")
                st.write(f"Issues: {', '.join(review.get('issues', []))}")
                if review.get("improvement_instructions"):
                    st.warning(review["improvement_instructions"])

    # Download
    st.divider()
    st.download_button(
        "⬇️ Download Research Brief (JSON)",
        data=json.dumps(brief, indent=2),
        file_name="research_brief.json",
        mime="application/json",
        use_container_width=True,
    )