import os
import re
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_MODEL = "llama3.2:3b"

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.2,
)


# ============================================================
# IN-MEMORY KNOWLEDGE BASE
# ============================================================

DOCUMENT_CHUNKS: List[Dict[str, Any]] = []


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200
) -> List[str]:

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(0, end - chunk_overlap)

    return chunks


# ============================================================
# TOKENIZE
# ============================================================

def tokenize(text: str) -> List[str]:
    return [
        word.lower()
        for word in re.findall(r"\b[a-zA-Z0-9]+\b", text)
        if len(word) > 2
    ]


# ============================================================
# RELEVANCE SEARCH
# ============================================================

def calculate_relevance(query: str, text: str) -> int:

    query_words = set(tokenize(query))
    text_words = set(tokenize(text))

    if not query_words:
        return 0

    common_words = query_words.intersection(text_words)

    score = len(common_words)

    query_lower = query.lower().strip()
    text_lower = text.lower()

    # Exact phrase bonus
    if query_lower and query_lower in text_lower:
        score += 10

    # Important name matching
    for word in query_words:
        if word in text_lower:
            score += 1

    return score


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(pdf_path: str) -> Dict[str, Any]:

    global DOCUMENT_CHUNKS

    pdf_path = str(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    reader = PdfReader(pdf_path)

    if len(reader.pages) == 0:
        raise ValueError("The PDF contains no pages.")

    filename = os.path.basename(pdf_path)

    new_chunks = []

    for page_number, page in enumerate(reader.pages):

        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        page_text = clean_text(page_text)

        if not page_text:
            continue

        chunks = split_text(page_text)

        for chunk_number, chunk in enumerate(chunks):

            new_chunks.append(
                {
                    "content": chunk,
                    "source": filename,
                    "page": page_number,
                    "page_label": str(page_number + 1),
                    "chunk": chunk_number,
                }
            )

    if not new_chunks:
        raise ValueError(
            "Could not extract readable text from the PDF."
        )

    # Remove previous version of the same document
    DOCUMENT_CHUNKS = [
        item
        for item in DOCUMENT_CHUNKS
        if item["source"] != filename
    ]

    # Add new chunks
    DOCUMENT_CHUNKS.extend(new_chunks)

    return {
        "message": "PDF uploaded and processed successfully",
        "filename": filename,
        "pages": len(reader.pages),
        "chunks": len(new_chunks),
    }


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(
    query: str,
    k: int = 4
) -> List[Dict[str, Any]]:

    if not isinstance(query, str):
        raise ValueError("query must be a string")

    if not isinstance(k, int):
        k = 4

    if k <= 0:
        k = 4

    if not DOCUMENT_CHUNKS:
        return []

    scored_results = []

    for document in DOCUMENT_CHUNKS:

        score = calculate_relevance(
            query,
            document["content"]
        )

        scored_results.append(
            {
                "content": document["content"],
                "source": document["source"],
                "page": document["page"],
                "page_label": document["page_label"],
                "score": score,
            }
        )

    scored_results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_results[:k]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results: List[Dict[str, Any]]
) -> str:

    if not results:
        return (
            "No relevant information was found in the "
            "uploaded documents."
        )

    context_parts = []

    for index, result in enumerate(results, start=1):

        context_parts.append(
            f"""
SOURCE {index}
File: {result["source"]}
Page: {result["page_label"]}

Content:
{result["content"]}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(
    query: str = None,
    question: str = None,
    k: int = 4
) -> Dict[str, Any]:

    if question is None:
        question = query

    if not question:
        raise ValueError("Question is required.")

    question = str(question).strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    # IMPORTANT:
    # k remains an integer.
    # This prevents the previous error where the question
    # accidentally became the value of k.

    results = search_documents(
        query=question,
        k=k
    )

    context = build_context(results)

    # ========================================================
    # RAG PROMPT
    # ========================================================

    prompt = f"""
You are an Enterprise RAG AI Assistant.

Your job is to answer the user's question using ONLY
the information contained in the DOCUMENT CONTEXT.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the documents,
   say clearly that the information was not found
   in the uploaded documents.
4. Give a concise and professional answer.
5. If useful, organize the answer using bullet points.
6. Never mention that you are guessing.
7. Base your answer strictly on the retrieved context.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    try:

        response = llm.invoke(prompt)

        answer = response.content

        if not answer:
            answer = (
                "I could not generate an answer from the "
                "uploaded documents."
            )

    except Exception as e:

        raise RuntimeError(
            f"AI question failed: {str(e)}"
        )

    # ========================================================
    # SOURCES
    # ========================================================

    sources = []

    seen = set()

    for result in results:

        source_key = (
            result["source"],
            result["page"]
        )

        if source_key in seen:
            continue

        seen.add(source_key)

        sources.append(
            {
                "source": result["source"],
                "page": result["page"],
                "page_label": result["page_label"],
            }
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


# ============================================================
# CLEAR DOCUMENTS
# ============================================================

def clear_documents():

    global DOCUMENT_CHUNKS

    DOCUMENT_CHUNKS = []


# ============================================================
# DOCUMENT STATUS
# ============================================================

def get_document_status() -> Dict[str, Any]:

    files = set(
        item["source"]
        for item in DOCUMENT_CHUNKS
    )

    return {
        "documents": len(files),
        "chunks": len(DOCUMENT_CHUNKS),
    }