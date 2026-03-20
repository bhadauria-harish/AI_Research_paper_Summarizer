import json
import tempfile

import streamlit as st

from pypdf2 import load_pdf
from workflow import run_analysis

st.set_page_config(page_title="Research Paper Analyzer", page_icon="📄", layout="wide")
st.title("📄 AI Research Paper Analyzer")
st.caption("Multi-agent system · LangGraph + Groq (llama-3.3-70b)")

# ── API Key handling ────────────────────────────
# Key lives only in st.session_state (per browser tab).
# os.environ is never touched — safe for shared/deployed servers.
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = ""

if not st.session_state["groq_api_key"]:
    st.warning("🔑 Groq API key not found. Please enter it below to continue.")
    typed_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Used only for this browser session. Never stored or shared.",
    )
    if typed_key:
        st.session_state["groq_api_key"] = typed_key
        st.rerun()
    else:
        st.stop()

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.success("✅ API key active for this session")
    if st.button("🔑 Change API Key"):
        st.session_state["groq_api_key"] = ""
        st.rerun()
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
            final_state = run_analysis(paper_text, api_key=st.session_state['groq_api_key'])
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

    def generate_pdf(brief):
        from fpdf import FPDF

        meta      = brief.get("paper_metadata", {})
        analysis  = brief.get("research_analysis", {})
        summary   = brief.get("executive_summary", "")
        citations = brief.get("citations_and_references", {})
        insights  = brief.get("key_insights", {})
        scores    = brief.get("quality_scores", {})

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        def title(text):
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_fill_color(230, 240, 255)
            pdf.cell(0, 10, text, ln=True, fill=True)
            pdf.ln(2)

        def field(label, value):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(45, 7, f"{label}:", ln=False)
            pdf.set_font("Helvetica", "", 10)
            safe = str(value).encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 7, safe)

        def bullet(text):
            pdf.set_font("Helvetica", "", 10)
            safe = str(text).encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 7, f"  - {safe}")

        def body(text):
            pdf.set_font("Helvetica", "", 10)
            safe = str(text).encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 7, safe)
            pdf.ln(2)

        # Header
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "Research Brief", ln=True, align="C")
        pdf.ln(4)

        # Metadata
        title("Paper Metadata")
        field("Title",   meta.get("title",   "Unknown"))
        field("Authors", ", ".join(meta.get("authors", [])))
        field("Year",    meta.get("year",    "Unknown"))
        field("Venue",   meta.get("venue",   "Unknown"))
        pdf.ln(4)

        # Analysis
        title("Research Analysis")
        field("Problem",     analysis.get("problem_statement", ""))
        field("Hypothesis",  analysis.get("hypothesis",        ""))
        field("Methodology", analysis.get("methodology",       ""))
        field("Experiments", analysis.get("experiments",       ""))
        field("Limitations", analysis.get("limitations",       ""))
        field("Future Work", analysis.get("future_work",       ""))
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Key Findings:", ln=True)
        for f in analysis.get("key_findings", []):
            bullet(f)
        pdf.ln(4)

        # Summary
        title("Executive Summary")
        body(summary)
        pdf.ln(2)

        # Citations
        title(f"Citations & References  (Total: {citations.get('total_references', 0)})")
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Key Related Works:", ln=True)
        for kw in citations.get("key_related_works", []):
            bullet(kw)
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "All References:", ln=True)
        for ref in citations.get("references", []):
            line = f"[{ref.get('index','?')}] {ref.get('authors','?')} ({ref.get('year','?')}). {ref.get('title','?')}. {ref.get('venue','')}"
            bullet(line)
        pdf.ln(4)

        # Insights
        title("Key Insights")
        field("Target Audience",    insights.get("target_audience",   ""))
        field("Difficulty Level",   insights.get("difficulty_level",  ""))
        field("Field Implications", insights.get("field_implications",""))
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Practical Takeaways:", ln=True)
        for t in insights.get("practical_takeaways", []):
            bullet(t)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Potential Applications:", ln=True)
        for a in insights.get("potential_applications", []):
            bullet(a)
        pdf.ln(4)

        # Quality Scores
        title("Quality Scores")
        for task, info in scores.items():
            status = "PASS" if info.get("passed") else "FAIL"
            field(task, f"{info.get('score')}/10  [{status}]")

        return bytes(pdf.output())

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download as JSON",
            data=json.dumps(brief, indent=2),
            file_name="research_brief.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        pdf_bytes = generate_pdf(brief)
        st.download_button(
            "⬇️ Download as PDF",
            data=pdf_bytes,
            file_name="research_brief.pdf",
            mime="application/pdf",
            use_container_width=True,
        )