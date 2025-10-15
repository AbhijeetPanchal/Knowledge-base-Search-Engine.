from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn
import os
import asyncio
from ingest import ingest_pdf, load_vectorstore, query_answer
from pathlib import Path

app = FastAPI()
DATA_DIR = Path(__file__).parent / "data"
VSTORE_PATH = DATA_DIR / "vstore.faiss"

@app.get("/", response_class=HTMLResponse)
async def home():
    html = (Path(__file__).parent / "frontend" / "index.html").read_text()
    return HTMLResponse(content=html)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    # ingest immediately
    await ingest_pdf(str(file_path), str(VSTORE_PATH))
    return {"status":"uploaded", "filename": file.filename}

@app.post("/query")
async def query(q: str = Form(...)):
    if not VSTORE_PATH.exists():
        return JSONResponse({"error":"No documents indexed yet. Upload PDFs first."}, status_code=400)
    answer = await query_answer(q, str(VSTORE_PATH))
    return {"query": q, "answer": answer}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
