import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Set dummy API key before importing the module that initializes Firecrawl
os.environ["FIRECRAWL_API_KEY"] = "fake_key"

# Add src to sys.path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.firecrawl_form_scraper import extract_structured_data, TravelDestination

@patch('src.firecrawl_form_scraper.app')
def test_extract_structured_data(mock_app):
    # Mock the result of app.scrape
    mock_result = MagicMock()
    mock_result.json = {
        "name": "Tatry",
        "location": "Poland",
        "description": "High mountains in Poland",
        "best_time_to_visit": "Summer",
        "attractions": ["Morskie Oko"],
        "difficulty_level": "Medium",
        "duration_days": 5
    }
    mock_app.scrape.return_value = mock_result

    result = extract_structured_data(
        url="https://wikivoyage.org",
        search_term="Tatry",
        input_selector="searchInput",
        schema_model=TravelDestination
    )

    assert result.json["name"] == "Tatry"
    assert result.json["location"] == "Poland"
    mock_app.scrape.assert_called_once()
