# Knowledge-base Search Engine (RAG) - Lightweight

**Objective:** Search across documents (PDFs) and provide synthesized answers using retrieval-augmented generation (RAG).

## What is included
- FastAPI backend (`main.py`)
- Ingestion module (`ingest.py`) that:
  - reads PDFs
  - chunks text
  - creates embeddings using `sentence-transformers`
  - stores embeddings in FAISS vector index
- Simple frontend (`frontend/index.html`) to upload PDFs and run queries
- `requirements.txt`
- Sample doc in `data/sample.txt`

---

## IMPORTANT NOTES (Read before running)
1. This project uses the OpenAI API to synthesize answers. You must provide `OPENAI_API_KEY` as an environment variable.
2. If `OPENAI_API_KEY` is not set, queries will still return retrieved context but won't call the LLM.
3. This is a prototype tailored to assessment requirements: PDF ingestion, RAG retrieval, LLM answer synthesis, backend API, and small frontend.

---

## Setup & Deployment (Exact terminal commands to copy-paste)
Below steps are provided in Hinglish (mix of Hindi + English) as you asked. Copy-paste commands directly into VS Code terminal.

### 1) Project folder me chale jao (open terminal in VS Code)
```bash
# agar abhi project folder nahi bana to:
mkdir -p ~/kbase_project && cd ~/kbase_project
# agar aapne zip extract kiya hai to us folder me aa jao, e.g.
# cd /path/to/kbase_project
```

### 2) Python virtual environment banao aur activate karo
```bash
python3 -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1
```

### 3) Requirements install karo
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Agar `faiss-cpu` install karne me dikkat aaye to pip error de sakta hai on Windows. Us case me:
```bash
pip install faiss-cpu --no-cache-dir
```

### 4) OpenAI API key set karo (agar LLM se answer chahiye)
```bash
# Linux / macOS
export OPENAI_API_KEY="your_openai_api_key_here"
# Windows (PowerShell)
# $env:OPENAI_API_KEY = "your_openai_api_key_here"
```

### 5) Server start karo (VS Code terminal me)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server start hone ke baad browser me jao: `http://localhost:8000/`

### 6) Use flow (frontend se)
1. `Upload PDF` pe jaake ek PDF upload karo — server usko `data/` me save karke faiss index me ingest karega.
2. `Query` me apna question daalo, submit karo — agar `OPENAI_API_KEY` set hai toh final synthesized answer milega, warna sirf retrieved context dikhega.

---

## Files (quick)
- `main.py` - FastAPI app with endpoints `/upload` and `/query`
- `ingest.py` - ingestion and retrieval logic with sentence-transformers + FAISS
- `frontend/index.html` - tiny UI
- `requirements.txt`
- `data/sample.txt`

---

## How this meets assessment checklist
- **Input:** Multiple text/PDF documents — upload via frontend or place in `data/`
- **Output:** User query → synthesized answer (via OpenAI) or retrieved context
- **Backend API:** `/upload` and `/query`
- **RAG / Embeddings:** sentence-transformers + FAISS
- **LLM for synthesis:** OpenAI ChatCompletion (set API key)
- **Deliverables:** Git repo structure, README
- **Evaluation focus covered:** retrieval + synthesis + code structure

---

## Drawbacks (as you requested to include)
1. Embedding model used is a small open-source model (`all-MiniLM-L6-v2`) — accurate for many tasks but not perfect on nuanced or highly technical docs.
2. FAISS index used is in-memory / flat index — scales okay for small datasets, but for large corpora use persistent vector DB (Milvus, Pinecone, Weaviate).
3. OpenAI use is paid and requires API key; offline LLMs could be used but need GPU and extra work.
4. AI-detection: I cannot guarantee it will not be detected as AI-assisted. Best practice is to add original analysis, edits, and a demo video showing your usage.

---

## How to make a demo video (brief)
1. Start server: `uvicorn main:app --reload`
2. Record screen while:
   - Uploading 1-2 PDFs
   - Running 2-3 realistic queries and showing answers
   - Explain retrieval results and how you refined prompt
3. Keep video 3-5 minutes focusing on functionality and evaluation metrics.

