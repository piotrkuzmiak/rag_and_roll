"""Deprecated module kept for compatibility.

Firecrawl integration has been removed from this project dependencies.
"""

from pydantic import BaseModel


class TravelDestination(BaseModel):
    name: str
    location: str
    description: str
    best_time_to_visit: str
    attractions: list[str]
    difficulty_level: str
    duration_days: int


def extract_structured_data(*args, **kwargs):
    raise RuntimeError(
        "Firecrawl support has been removed from this project. "
        "Install and integrate firecrawl-py manually if needed."
    )
