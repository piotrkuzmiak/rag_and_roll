from firecrawl import Firecrawl
from firecrawl.types import ClickAction, JsonFormat, Location, PressAction, WaitAction, WriteAction
from pydantic import BaseModel
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Initialize Firecrawl
app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

class TravelDestination(BaseModel):
    """Schema for extracting travel destination information."""
    name: str
    location: str
    description: str
    best_time_to_visit: str
    attractions: list[str]
    difficulty_level: str
    duration_days: int

def extract_structured_data(url: str, search_term: str,input_selector: str, schema_model=None, prompt=None):
    """
    Extract structured data from a URL using Firecrawl with JSON schema.

    Args:
        url: The URL to extract data from
        schema_model: Pydantic model class for structured extraction
        prompt: Custom prompt for the extraction
    """
    actions = [
        ClickAction(selector=input_selector),
        WriteAction(text=search_term),
        PressAction(key="Enter"),
        WaitAction(milliseconds=1500),
    ]
    # Convert Pydantic model to JSON schema
    schema = None
    if schema_model:
        schema = schema_model.model_json_schema()

    # Default prompt if none provided
    if not prompt and schema_model:
        prompt = f"Extract {schema_model.__name__} information from this webpage."

    # Use Firecrawl's JSON extraction
    result = app.scrape(
        url,
        actions=actions,
        formats=[JsonFormat(type="json", schema=schema, prompt=prompt)],
        only_main_content=True,
        timeout=30000,
        remove_base64_images=True,
        include_tags=['h1', 'h2', 'h3', 'p', 'li', 'a', 'div', 'span'],
        exclude_tags=['script', 'style', 'nav', 'footer', 'aside', 'video', 'img'],
        max_age=3600,
        block_ads=True,
        location=Location(country='PL', languages=['pl'])
    )

    return result

if __name__ == "__main__":
    print("=== Example 1: Extract Travel Destination Info ===")
    travel_url = "https://wikivoyage.org"
    travel_result = extract_structured_data(
        url=travel_url,
        search_term="Tatry",
        input_selector="#searchInput",
        schema_model=TravelDestination,
        prompt="Extract information about this travel destination including name, location, description, best time to visit, main attractions, difficulty level, and typical duration."
    )

    if travel_result.json:
        print("Extracted Travel Data:")
        print(travel_result.json)
    else:
        print("Failed to extract travel data or no JSON returned")