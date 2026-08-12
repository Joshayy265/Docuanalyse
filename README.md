# DocuAnalyse — Ask Questions About Your PDFs

A retrieval-augmented generation (RAG) app that lets you upload a set of PDFs and
then ask questions about them in a chat interface. Every answer comes back with
the source document and page number it was drawn from, so claims can be checked
rather than trusted.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-0467DF)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Demo

📹 **[Watch the demo video](../../releases/latest)** — download `Docuanalyse.mp4`
from the latest release.

---

## Why this exists

A language model asked about a document it has never seen will invent a
plausible answer. RAG fixes that by retrieving the relevant passages first and
instructing the model to answer only from them. This project is a compact,
readable implementation of that pipeline — roughly 200 lines end to end — built
to understand each stage rather than to wrap a framework.

The design choice worth pointing at: **the answer is worth little without its
provenance**. `return_source_documents=True` is what makes the Sources panel
possible, and that panel is the difference between a demo and something you'd
actually rely on.

---

## How it works

```
PDFs uploaded via Streamlit sidebar
        │
        ▼
  PyPDFLoader                     one LangChain Document per page,
        │                         tagged with metadata["source"] = filename
        ▼
  RecursiveCharacterTextSplitter  chunk_size=1000, chunk_overlap=150
        │                         overlap keeps sentences from being cut
        ▼                         mid-thought at a boundary
  HuggingFace all-MiniLM-L6-v2    384-dim sentence embeddings, CPU
        │
        ▼
  FAISS (in-memory index)         cosine similarity over the chunks
        │
        ▼
  RetrievalQA + ChatGroq          llama-3.1-8b-instant answers from
        │                         the retrieved chunks only
        ▼
  Answer + Sources expander       filename and page for every chunk used
```

### Design decisions

| Choice | Why |
|---|---|
| **all-MiniLM-L6-v2** | 384-dim, ~80 MB, runs on CPU in milliseconds. Larger embedding models improve recall marginally but need a GPU to stay interactive. |
| **FAISS in-memory** | The index is rebuilt per session, so there is no stale-state or persistence bug surface. Fine up to a few hundred pages; a persistent store (Chroma, pgvector) is the next step. |
| **Chunk 1000 / overlap 150** | Large enough to hold a complete argument, small enough that retrieval stays precise. The 15% overlap prevents a sentence split across a boundary from becoming unretrievable. |
| **Groq + Llama 3.1 8B** | Groq's inference latency makes chat feel responsive; the 8B model is sufficient when the context is supplied by retrieval rather than recalled from weights. |

### Analysis modes

The sidebar mode prepends an instruction to the query before retrieval:

- **General Q&A** — the question is passed through unchanged
- **Executive Summary** — `"Provide an executive summary: " + question`
- **Action Items** — `"Extract action items and decisions: " + question`

---

## Running it locally

**Prerequisites:** Python 3.10+ and a Groq API key ([console.groq.com](https://console.groq.com), free tier available).

```bash
git clone https://github.com/Joshayy265/Docuanalyse.git
cd Docuanalyse
pip install -r requirements.txt

cp .env.example .env       # then paste your key into .env
streamlit run app.py
```

Open http://localhost:8501, upload one or more PDFs in the sidebar, and ask away.

The first run downloads the MiniLM embedding model (~80 MB) from HuggingFace.

---

## Project structure

```
Docuanalyse/
├── app.py             # Entire pipeline: UI, loading, chunking, index, chat
├── requirements.txt
├── .env.example       # GROQ_API_KEY placeholder
├── LICENSE
└── README.md
```

---

## Known limitations

- **PDFs only.** `PyPDFLoader` is the sole loader; `.docx`, `.txt` and `.md`
  would each need their own.
- **Scanned PDFs return nothing.** There is no OCR step, so image-only pages
  produce empty text and silently contribute no chunks.
- **The index is rebuilt on every rerun.** Streamlit re-executes the script on
  each interaction, so re-embedding happens more often than it needs to.
  Wrapping the index in `@st.cache_resource` is the obvious fix.
- **Uploaded files are written to the working directory** as `temp_<name>.pdf`
  and never cleaned up.
- **No conversational memory.** Each question is retrieved and answered
  independently, so follow-ups like "what about the second one?" lose context.

---

## License

MIT — see [LICENSE](LICENSE).
