import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag import (
    process_pdf,
    ask_question,
    get_document_status,
    clear_documents,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Enterprise RAG AI",
    description="Local document intelligence using RAG and Ollama",
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):

    question: str

    k: int = 4


# ============================================================
# FRONTEND
# ============================================================

@app.get(
    "/app",
    response_class=HTMLResponse
)
async def web_app():

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail="frontend/index.html not found"
        )

    return index_file.read_text(
        encoding="utf-8"
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "name": "Enterprise RAG AI",
        "status": "online",
        "app": "/app",
        "docs": "/docs",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "online",
        "message": "Enterprise RAG AI is running"
    }


# ============================================================
# UPLOAD PDF
# ============================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    filename = Path(file.filename).name

    if not filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    save_path = DATA_DIR / filename

    try:

        # Save uploaded file to disk
        with save_path.open("wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # IMPORTANT:
        # process_pdf expects a FILE PATH.
        # We pass the saved path, not UploadFile.
        result = process_pdf(
            str(save_path)
        )

        return {
            "success": True,
            **result
        }

    except Exception as e:

        if save_path.exists():

            try:
                save_path.unlink()
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )

    finally:

        await file.close()


# ============================================================
# ASK AI
# ============================================================

@app.post("/ask")
async def ask_ai(
    request: QuestionRequest
):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    status = get_document_status()

    if status["documents"] == 0:

        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF before asking a question."
        )

    try:

        result = ask_question(
            question=question,
            k=request.k
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"AI question failed: {str(e)}"
        )


# ============================================================
# STATUS
# ============================================================

@app.get("/status")
async def status():

    return get_document_status()


# ============================================================
# CLEAR KNOWLEDGE BASE
# ============================================================

@app.delete("/documents")
async def delete_documents():

    clear_documents()

    # Remove PDFs from data folder
    for file in DATA_DIR.iterdir():

        if file.is_file() and file.suffix.lower() == ".pdf":

            try:
                file.unlink()
            except Exception:
                pass

    return {
        "success": True,
        "message": "Knowledge base cleared."
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    print("=" * 60)
    print("Enterprise RAG AI")
    print("=" * 60)
    print("Server started successfully.")
    print(f"Data directory: {DATA_DIR}")
    print(f"Frontend directory: {FRONTEND_DIR}")
    print("Ollama model: llama3.2:3b")
    print("=" * 60)