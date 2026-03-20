# 📄 AI Research Paper Analyzer

A **multi-agent system** that automatically reads, analyzes, and summarizes academic research papers.
Built with **LangGraph** for orchestration, **Groq** (llama-3.3-70b-versatile) as the LLM, and **LangSmith** for tracing.

---

## 🏗️ Architecture

```
Upload PDF
     │
     ▼
pdf_loader.py  ──►  raw paper text
     │
     ▼
LangGraph Workflow (workflow.py)
     │
     ├──► Paper Analyzer Agent  ──► Review Agent ──► [pass / retry x2]
     │         (agents.py)
     ├──► Summary Generator     ──► Review Agent ──► [pass / retry x2]
     │
     ├──► Citation Extractor    ──► Review Agent ──► [pass / retry x2]
     │
     ├──► Key Insights Agent    ──► Review Agent ──► [pass / retry x2]
     │
     └──► Boss Combiner  ──►  Final Research Brief
```

### Agents (all in `agents.py`)

| Agent | What it does |
|-------|-------------|
| **Paper Analyzer** | Extracts title, authors, methodology, hypothesis, key findings, limitations |
| **Summary Generator** | Writes a 150–200 word executive summary |
| **Citation Extractor** | Pulls out all references and highlights key related works |
| **Key Insights** | Practical takeaways, applications, difficulty level, target audience |
| **Review Agent** | Scores every output 1–10. Retries if score < 7 (max 2 retries) |
| **Boss Combiner** | Merges all outputs into the final research brief |

---

## 📂 Project Structure

```
research_analyzer/
├── agents.py          # All 5 agents — LangChain chains + JsonOutputParser
├── workflow.py        # LangGraph state machine + retry logic + LangSmith tracing
├── pdf_loader.py      # PDF text extraction using PyPDF2
├── app.py             # Streamlit web UI
├── main.py            # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Setup

### 1. Clone & install

```bash
git clone https://github.com/bhadauria-harish/AI_Research_paper_Summarizer.git
cd research-paper-analyzer

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Add API keys

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=gsk_...

# Optional — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=research-paper-analyzer
```

> If you don't have a `.env` file, the Streamlit app will ask for your Groq API key on screen.

---

## ▶️ Running the App

### Web UI (recommended)
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501), upload a PDF, click **Analyze Paper**.

### CLI
```bash
python main.py --source paper.pdf
python main.py --source paper.pdf --output my_brief.json
```

---

## 📤 Output

The system generates a **Research Brief** downloadable as **JSON** or **PDF** with:

- **Paper Metadata** — title, authors, year, venue
- **Research Analysis** — problem, hypothesis, methodology, experiments, findings, limitations
- **Executive Summary** — 150–200 word overview
- **Citations & References** — full reference list + key related works
- **Key Insights** — takeaways, applications, difficulty level, target audience
- **Quality Scores** — per-agent review scores (1–10)

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Get free at [console.groq.com](https://console.groq.com) |
| `LANGCHAIN_TRACING_V2` | ❌ Optional | Set to `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | ❌ Optional | Get at [smith.langchain.com](https://smith.langchain.com) |
| `LANGCHAIN_PROJECT` | ❌ Optional | Project name in LangSmith dashboard |

---

## 🧪 Sample Papers to Test

- **Attention Is All You Need** — https://arxiv.org/abs/1706.03762
- **BERT** — https://arxiv.org/abs/1810.04805
- **GPT-3** — https://arxiv.org/abs/2005.14165

> Download the PDF from arXiv and upload it directly to the app.

---

## ⚠️ Known Limitations

- Scanned / image-based PDFs are not supported — the PDF must have selectable text
- Papers longer than ~40,000 characters are truncated before sending to the LLM
- Citation extraction depends on how well-formatted the reference section is

---

## 📝 License

MIT