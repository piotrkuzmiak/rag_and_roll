# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run RAG CLI (ChromaDB-based)
uv run python main.py

# Run Wikiloc trail planning agent
uv run python wikiloc_agent.py

# Force re-index ChromaDB from CSV
uv run python -c "import main; main.main(force_reindex=True)"

# Lint
uv run ruff check .

# Run tests
uv run pytest tests/
```

## Architecture

Two independent agent applications built on **Pydantic AI** with **OpenAI** models, sharing no state.

### Application 1: ChromaDB RAG (`main.py`)

Answers Polish mountain trail questions from a local vector database.

- **Indexing** (`src/textfile_to_chromadb.py`): Reads `polish_mountains_hiking_trails_fake.csv` in 500-row chunks, embeds with `all-MiniLM-L6-v2` (CUDA if available), persists to `./chroma_storage/`.
- **Agent**: `openai:gpt-4-nano`. Deps pattern: `RAGDeps` dataclass injects the ChromaDB collection. One tool `search_hiking_trails()` does semantic search and returns formatted context (description + address, GPS, hours metadata).

### Application 2: Wikiloc Trail Planner (`wikiloc_agent.py`)

Plans hiking itineraries using live data from Wikiloc.

- **Scraper** (`src/wikiloc_scraper.py`): Uses Firecrawl to load a Wikiloc page (for session cookies + Cloudflare bypass), then injects a JS `fetch()` call to Wikiloc's internal `find.do` API. The API response is written to a hidden DOM element (`#__wl_api_result`), extracted from raw HTML, and parsed directly — no LLM extraction for search. Trail detail pages still use Firecrawl's LLM JSON extraction.
- **Internal search API**: `GET /wikiloc/find.do?event=map&to=24&sw=-89.999,-179.999&ne=89.999,179.999&q=<query>` returns `{count, spas[]}`. Key fields: `prettyURL`, `name`, `pictoText`, `near`, `rating`, `numRatings`, `distance` (in `uom` units — miles or km), `slope` (elevation in `uomslope` units — feet or meters), `skill` (1–4 mapped to Easy/Moderate/Hard/Expert). Unit conversion happens in `_parse_trail_summary()`.
- **Agent** (`wikiloc_agent.py`): `openai:gpt-4o-mini`. Agent and tools created inside `build_agent()` (deferred init to avoid OpenAI key check at import time). Three tools: `search_wikiloc_trails`, `get_wikiloc_trail_details`, `find_trails_with_full_details`.

### Pydantic AI pattern

Both agents follow the same structure: `@dataclass` deps type → `Agent(model, system_prompt, deps_type=...)` → `@agent.tool` functions receiving `RunContext[DepsType]` as first arg (required by framework even when unused) → `agent.run_sync(query, deps=deps)`.

### Environment variables

Both agents load `.env` automatically via `python-dotenv`.

- `OPENAI_API_KEY` — required by both (prompted at runtime if absent)
- `FIRECRAWL_API_KEY` — required by Wikiloc agent
