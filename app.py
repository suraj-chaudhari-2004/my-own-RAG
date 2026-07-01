from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import requests
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


APP_TITLE = "PDF RAG"
INDEX_DIR = Path(".rag_index")
INDEX_FILE = INDEX_DIR / "index.json"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
PDF_DIR = INDEX_DIR / "pdfs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class Chunk:
    id: str
    source: str
    page: int
    text: str


def load_dotenv() -> None:
    env_file = Path(".env")
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def config_value(section: str, key: str, env_key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(section, {}).get(key)
    except Exception:
        value = None
    return str(value or os.getenv(env_key, default))


@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def ensure_storage() -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 900, overlap: int = 140) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_chunks(pdf_path: Path, source_name: str, source_id: str) -> list[Chunk]:
    reader = PdfReader(str(pdf_path))
    chunks: list[Chunk] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        for chunk_index, chunk_text in enumerate(split_text(text)):
            chunks.append(
                Chunk(
                    id=f"{source_id}:p{page_number}:c{chunk_index}",
                    source=source_name,
                    page=page_number,
                    text=chunk_text,
                )
            )
    return chunks


def load_index() -> tuple[list[Chunk], np.ndarray | None]:
    if not INDEX_FILE.exists() or not EMBEDDINGS_FILE.exists():
        return [], None

    raw_chunks = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in raw_chunks]
    embeddings = np.load(EMBEDDINGS_FILE)
    return chunks, embeddings


def save_index(chunks: list[Chunk], embeddings: np.ndarray) -> None:
    ensure_storage()
    INDEX_FILE.write_text(
        json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(EMBEDDINGS_FILE, embeddings)


def embed_texts(texts: Iterable[str]) -> np.ndarray:
    vectors = get_embedder().encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def add_pdfs(uploaded_files: list) -> int:
    ensure_storage()
    chunks, embeddings = load_index()
    existing_source_ids = {chunk.id.split(":")[0] for chunk in chunks}
    new_chunks: list[Chunk] = []

    for uploaded_file in uploaded_files:
        data = uploaded_file.getvalue()
        source_id = file_hash(data)
        if source_id in existing_source_ids:
            continue

        pdf_path = PDF_DIR / f"{source_id}_{uploaded_file.name}"
        pdf_path.write_bytes(data)
        new_chunks.extend(extract_pdf_chunks(pdf_path, uploaded_file.name, source_id))

    if not new_chunks:
        return 0

    new_embeddings = embed_texts(chunk.text for chunk in new_chunks)
    all_chunks = chunks + new_chunks
    all_embeddings = (
        new_embeddings
        if embeddings is None
        else np.vstack([embeddings, new_embeddings]).astype(np.float32)
    )
    save_index(all_chunks, all_embeddings)
    return len(new_chunks)


def retrieve(question: str, top_k: int) -> list[tuple[Chunk, float]]:
    chunks, embeddings = load_index()
    if not chunks or embeddings is None:
        return []

    query_embedding = embed_texts([question])[0]
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[index], float(scores[index])) for index in top_indices]


def build_prompt(question: str, matches: list[tuple[Chunk, float]]) -> str:
    context_blocks = []
    for index, (chunk, score) in enumerate(matches, start=1):
        context_blocks.append(
            f"[{index}] Source: {chunk.source}, page {chunk.page}, score {score:.3f}\n"
            f"{chunk.text}"
        )

    context = "\n\n".join(context_blocks)
    return (
        "Answer the question using only the provided PDF context. "
        "If the answer is not present, say you could not find it in the PDFs. "
        "Cite sources with bracket numbers like [1] and mention page numbers when useful.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )


def call_openrouter(prompt: str, model: str, api_key: str, site_url: str, app_name: str) -> str:
    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        },
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer questions using supplied PDF context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def call_ollama(prompt: str, model: str, base_url: str) -> str:
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def clear_index() -> None:
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
    if EMBEDDINGS_FILE.exists():
        EMBEDDINGS_FILE.unlink()
    if PDF_DIR.exists():
        for pdf_path in PDF_DIR.glob("*.pdf"):
            pdf_path.unlink()


def render_sidebar() -> dict:
    st.sidebar.header("Model")
    provider = st.sidebar.radio("LLM call", ["OpenRouter API", "Local Ollama"])
    top_k = st.sidebar.slider("Retrieved chunks", min_value=2, max_value=10, value=5)

    config = {"provider": provider, "top_k": top_k}
    if provider == "OpenRouter API":
        config["api_key"] = st.sidebar.text_input(
            "OpenRouter API key",
            value=config_value("openrouter", "api_key", "OPENROUTER_API_KEY"),
            type="password",
        )
        config["model"] = st.sidebar.text_input(
            "OpenRouter model",
            value=config_value("openrouter", "model", "OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )
        config["site_url"] = st.sidebar.text_input(
            "Site URL",
            value=config_value("openrouter", "site_url", "OPENROUTER_SITE_URL", "http://localhost:8501"),
        )
        config["app_name"] = st.sidebar.text_input(
            "App name",
            value=config_value("openrouter", "app_name", "OPENROUTER_APP_NAME", "My PDF RAG"),
        )
    else:
        config["base_url"] = st.sidebar.text_input(
            "Ollama URL",
            value=config_value("ollama", "base_url", "OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        config["model"] = st.sidebar.text_input(
            "Ollama model",
            value=config_value("ollama", "model", "OLLAMA_MODEL", "llama3.1"),
        )

    st.sidebar.divider()
    if st.sidebar.button("Clear PDF index", use_container_width=True):
        clear_index()
        st.sidebar.success("Index cleared.")
        st.rerun()

    return config


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    config = render_sidebar()
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process PDFs", type="primary"):
        with st.spinner("Extracting text and building embeddings..."):
            added_count = add_pdfs(uploaded_files)
        if added_count:
            st.success(f"Added {added_count} searchable chunks.")
        else:
            st.info("No new PDF content was added.")

    chunks, _ = load_index()
    source_names = sorted({chunk.source for chunk in chunks})
    st.caption(
        f"Indexed chunks: {len(chunks)}"
        + (f" across {len(source_names)} PDF(s): {', '.join(source_names)}" if source_names else "")
    )

    question = st.chat_input("Ask a question about your PDFs")
    if not question:
        return

    if not chunks:
        st.warning("Upload and process at least one PDF before asking questions.")
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant PDF sections..."):
            matches = retrieve(question, int(config["top_k"]))
            prompt = build_prompt(question, matches)

        try:
            with st.spinner("Calling model..."):
                if config["provider"] == "OpenRouter API":
                    if not config.get("api_key"):
                        st.error("Enter an OpenRouter API key in the sidebar or set OPENROUTER_API_KEY.")
                        return
                    answer = call_openrouter(
                        prompt=prompt,
                        model=str(config["model"]),
                        api_key=str(config["api_key"]),
                        site_url=str(config["site_url"]),
                        app_name=str(config["app_name"]),
                    )
                else:
                    answer = call_ollama(
                        prompt=prompt,
                        model=str(config["model"]),
                        base_url=str(config["base_url"]),
                    )
        except Exception as exc:
            st.error(f"Model call failed: {exc}")
            return

        st.write(answer)

        with st.expander("Retrieved sources"):
            for index, (chunk, score) in enumerate(matches, start=1):
                st.markdown(f"**[{index}] {chunk.source}, page {chunk.page}** - score `{score:.3f}`")
                preview = chunk.text[:1200]
                st.write(preview + ("..." if len(chunk.text) > len(preview) else ""))


if __name__ == "__main__":
    main()
