import os
from sentence_transformers import SentenceTransformer
import faiss
import json
from PyPDF2 import PdfReader
from pathlib import Path
import numpy as np
import pickle
import asyncio
import openai
from typing import List

EMBED_MODEL = "all-MiniLM-L6-v2"

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for p in reader.pages:
        try:
            texts.append(p.extract_text() or "")
        except Exception:
            pass
    return "\n".join(texts)

def chunk_text(text: str, chunk_size:int=500, overlap:int=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

async def ingest_pdf(path: str, vstore_path: str):
    print("Ingesting", path)
    text = read_pdf_text(path)
    chunks = chunk_text(text)
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    if os.path.exists(vstore_path):
        # load existing index and add
        index = faiss.read_index(vstore_path)
        index.add(embeddings)
        faiss.write_index(index, vstore_path)
        meta_path = vstore_path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                metas = pickle.load(f)
        else:
            metas = []
        metas.extend([{"source": os.path.basename(path), "text": c} for c in chunks])
        with open(meta_path, "wb") as f:
            pickle.dump(metas, f)
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, vstore_path)
        metas = [{"source": os.path.basename(path), "text": c} for c in chunks]
        with open(vstore_path + ".meta", "wb") as f:
            pickle.dump(metas, f)
    print("Ingestion complete. Chunks:", len(chunks))

async def load_vectorstore(vstore_path: str):
    import pickle
    if not os.path.exists(vstore_path):
        raise FileNotFoundError("vstore not found")
    index = faiss.read_index(vstore_path)
    with open(vstore_path + ".meta", "rb") as f:
        metas = pickle.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    return index, metas, model

async def query_answer(query: str, vstore_path: str, k:int=4) -> str:
    # load
    index, metas, model = await load_vectorstore(vstore_path)
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    retrieved = [metas[i] for i in I[0] if i < len(metas)]
    # build context
    context = "\n---\n".join([f"Source: {r['source']}\nText: {r['text']}" for r in retrieved])
    prompt = f"Using the following documents, answer the user's question succinctly.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    # call OpenAI (user must set OPENAI_API_KEY in env)
    import os, openai
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "OPENAI_API_KEY not set. Set it in environment to get a synthesized answer. Retrieved context:\n" + context
    openai.api_key = key
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=300,
        temperature=0.2,
    )
    return resp["choices"][0]["message"]["content"].strip()
