# rag_and_roll

`rag_and_roll` is a simple Retrieval-Augmented Generation (RAG) demo that loads Wikivoyage data into ChromaDB and answers questions with Gemini.

## Installation

This project uses `uv` for dependency management:

```bash
uv sync
```

## Configuration

Set your Google API key:

```bash
export GOOGLE_API_KEY=your_google_api_key
```

Optional runtime config:

```bash
export GOOGLE_API_VERSION=v1beta
export GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
```

## Usage

Run:

```bash
python main.py
```
