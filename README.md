# PDF RAG App

A local PDF RAG application that answers questions from uploaded PDFs.

It supports two LLM modes:

- **OpenRouter API** using your OpenRouter API key.
- **Local Ollama** using a model running on your machine.

PDF text extraction and embeddings run locally. The searchable index is stored in `.rag_index/`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or install from `pyproject.toml`:

```powershell
pip install -e .
```

Copy the example environment file if you want saved defaults:

```powershell
Copy-Item .env.example .env
```

You can also copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`, which is the native Streamlit config file for secrets:

```toml
[openrouter]
api_key = "your_openrouter_api_key_here"
model = "openai/gpt-4o-mini"
```

If you use `.env`, set:

```text
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

## Run

```powershell
streamlit run app.py
```

## Test PDF

A sample PDF is available at `sample_pdfs/rag_test_document.pdf`.

Good test questions:

- Who owns Project Phoenix?
- Which API provider is approved?
- Which local model is recommended?
- What data should not be uploaded?

## Local LLM Option

Install Ollama, then pull a model:

```powershell
ollama pull llama3.1
ollama serve
```

In the app sidebar, choose **Local Ollama** and set the model name.
