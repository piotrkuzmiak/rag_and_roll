# rag_and_roll

`rag_and_roll` is a Retrieval-Augmented Generation (RAG) CLI demo for Polish mountain attractions.
It:

1. Loads `polish_mountains_hiking_trails_fake.csv` into a ChromaDB collection.
2. Uses `all-MiniLM-L6-v2` sentence-transformer embeddings (GPU if available, otherwise CPU).
3. Runs an agent that retrieves relevant records and answers questions using only retrieved context.

## Installation

This project uses `uv`:

```bash
uv sync
```

## Configuration

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

If `OPENAI_API_KEY` is not set, `main.py` will prompt for it at startup.

## Run

Start the interactive CLI:

```bash
python main.py
```

Type questions at the `Query:` prompt and press Enter.  
Use `Ctrl+C` to exit.

## Optional: force reindex

By default, the existing ChromaDB index is reused when available. To force rebuilding:

```bash
python -c "import main; main.main(force_reindex=True)"
```
