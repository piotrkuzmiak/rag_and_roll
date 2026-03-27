# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run RAG CLI (ChromaDB-based)
python main.py

# Run Wikiloc trail planning agent
python wikiloc_agent.py

# Force re-index ChromaDB from CSV
python -c "from src.textfile_to_chromadb import create_chromadb_collection_from_csv; create_chromadb_collection_from_csv(force_reindex=True)"

# Lint
uv run ruff check .

# Run tests
uv run pytest tests/
```

## Architecture

This is a RAG demo for Polish mountain attractions with two independent agent applications, both built on **Pydantic AI** with **OpenAI** models.

### Application 1: ChromaDB RAG (`main.py`)

Answers hiking trail questions using a local vector database indexed from `polish_mountains_hiking_trails_fake.csv`.

- **Indexing** (`src/textfile_to_chromadb.py`): Reads CSV in 500-row chunks, generates embeddings with `sentence-transformers/all-MiniLM-L6-v2` (GPU if CUDA available), persists to `./chroma_storage/`.
- **Agent** (`main.py`): Uses `openai:gpt-4-nano`. The `RAGDeps` dataclass injects the ChromaDB collection. The agent has one tool `search_hiking_trails()` that performs semantic search and returns descriptions + metadata (address, GPS, hours, etc.).

### Application 2: Wikiloc Agent (`wikiloc_agent.py`)

Plans hiking itineraries by scraping live data from Wikiloc via Firecrawl.

- **Scraper** (`src/wikiloc_scraper.py`): Uses Firecrawl (LLM-based extraction from HTML) to parse Wikiloc search results and trail detail pages. Handles GDPR popups via JavaScript injection. Returns typed Pydantic models (`TrailSummary`, `TrailDetail`).
- **Agent** (`wikiloc_agent.py`): Uses `openai:gpt-4o-mini`. Exposes three tools: `search_wikiloc_trails()` (fast, summaries only), `get_wikiloc_trail_details()` (full details for a URL), `find_trails_with_full_details()` (search + auto-enrich).

### Environment Variables

- `OPENAI_API_KEY` — required by both agents (prompted at runtime if not set)
- `FIRECRAWL_API_KEY` — required by Wikiloc agent (set in `.env`)
