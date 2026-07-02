import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import requests
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None

OLLAMA_URL = "http://localhost:11434/api"

@dataclass
class Chunk:
    id: str
    text: str
    source: str


def extract_text_from_file(path: str) -> Tuple[str, str]:
    """Extract text from PDF/TXT/image. Returns text and tool name."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".txt":
        return p.read_text(encoding="utf-8", errors="ignore"), "file_parsing"
    if suffix == ".pdf":
        if fitz is None:
            return "", "pdf_parser_missing"
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        return text, "pdf_parsing"
    if suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        if pytesseract is None:
            return "", "ocr_missing"
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang="eng")
        return text, "ocr"
    return "", "unsupported_file"


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120, source: str = "document") -> List[Chunk]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    i = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(Chunk(id=f"{source}_chunk_{i}", text=text[start:end], source=source))
        if end == len(text):
            break
        start = max(0, end - overlap)
        i += 1
    return chunks


class LocalVectorStore:
    """Simple TF-IDF vector store fallback. Good for demo and evaluation without cloud services."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words=None)
        self.chunks: List[Chunk] = []
        self.matrix = None

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts) if texts else None

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.matrix is None or not self.chunks:
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).flatten()
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk_id": self.chunks[i].id, "text": self.chunks[i].text, "score": float(scores[i]), "source": self.chunks[i].source}
            for i in order
        ]


def ollama_generate(prompt: str, model: str = "llama3.2") -> str:
    """Generate with Ollama. If Ollama is unavailable, returns a deterministic demo answer."""
    try:
        r = requests.post(f"{OLLAMA_URL}/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception:
        return "[DEMO FALLBACK] Ollama не е стартиран. В реалното демо тук се генерира отговор от локален LLM върху намерения контекст."


def ollama_vision(image_path: str, prompt: str = "Describe the image.", model: str = "llava") -> str:
    try:
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        payload = {
                    "model": model,
                    "prompt": prompt,
                    "images": [b64],
                    "stream": False,
                    "options": {
                                "temperature": 0.1,
                                "top_p": 0.8
    }
}
        r = requests.post(f"{OLLAMA_URL}/generate", json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception:
        return "[DEMO FALLBACK] Vision моделът не е достъпен. В реалното демо LLaVA/llama3.2-vision описва изображението локално."


def answer_with_rag(query: str, retrieved: List[Dict[str, Any]], model: str = "llama3.2") -> str:
    context = "\n\n".join([f"[{r['chunk_id']}] {r['text']}" for r in retrieved])
    prompt = f"""
Ти си локален AI агент за откриване на знания. Отговори само по предоставения контекст.

Контекст:
{context}

Въпрос: {query}

Отговори кратко, точно и структурирано на български.
"""
    return ollama_generate(prompt, model=model)


def evaluate_precision_recall(retrieved_chunk_ids: List[str], relevant_chunk_ids: List[str], k: int) -> Dict[str, float]:
    top = retrieved_chunk_ids[:k]
    relevant = set(relevant_chunk_ids)
    hits = sum(1 for c in top if c in relevant)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    return {"precision_at_k": precision, "recall_at_k": recall, "hits": hits}
