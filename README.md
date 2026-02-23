# rag_and_roll

`rag_and_roll` is a repository dedicated to developing LLM-based applications, specifically focusing on Retrieval-Augmented Generation (RAG) and structured data extraction. The project currently leverages [Firecrawl](https://www.firecrawl.dev/) to interact with web content and extract meaningful information.

## Features

- **Structured Web Scraping**: Utilize Firecrawl to perform complex web interactions (filling forms, searching) and extract data into structured JSON formats using Pydantic models.
- **Wikivoyage Integration**: Example implementation for extracting travel destination details from Wikivoyage.

## Installation

This project uses `uv` for dependency management. To get started, clone the repository and run:

```bash
uv sync
```

Alternatively, you can install the dependencies using `pip`:

```bash
pip install firecrawl-py pydantic-ai python-dotenv
```

## Configuration

The project requires a Firecrawl API key. Create a `.env` file in the root directory:

```env
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

## Usage

### Firecrawl Form Scraper

The `src/firecrawl_form_scraper.py` script demonstrates how to automate form interactions and extract structured data.

To run the example:

```bash
python src/firecrawl_form_scraper.py
```

#### How it works

The script defines an `extract_structured_data` function that:
1.  **Navigates** to the target URL.
2.  **Interacts** with the page by typing a search term into an input field and pressing "Enter".
3.  **Waits** for the content to load and takes a screenshot.
4.  **Extracts** structured data based on a provided Pydantic schema (e.g., `TravelDestination`).

Example Pydantic schema used in the script:

```python
class TravelDestination(BaseModel):
    name: str
    location: str
    description: str
    best_time_to_visit: str
    attractions: list[str]
    difficulty_level: str
    duration_days: int
```
