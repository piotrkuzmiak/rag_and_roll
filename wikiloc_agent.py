"""
Trail planning agent powered by Wikiloc.

Searches Wikiloc for real trails and helps design hiking itineraries for your trips.

Usage:
    python wikiloc_agent.py
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from src.wikiloc_scraper import (
    TrailDetail,
    TrailSearchResults,
    find_trails_with_details,
    get_trail_details,
    search_trails,
)

load_dotenv()

SYSTEM_PROMPT = (
    "You are a trail planning assistant that helps design hiking and outdoor itineraries for trips. "
    "You have access to real trail data from Wikiloc. "
    "\n\n"
    "Use the available tools to:\n"
    "- search_wikiloc_trails: quickly find trails by location/keyword (returns summaries)\n"
    "- get_wikiloc_trail_details: fetch full details for a specific trail URL\n"
    "- find_trails_with_full_details: search and enrich results with full details "
    "(use when the user wants recommendations with rich context, but keep max_trails low)\n"
    "\n"
    "Always use tools before answering — do not invent trail data. "
    "When designing an itinerary, consider: difficulty, distance, elevation gain, duration, "
    "route type (loop vs out-and-back), and the user's fitness level if mentioned. "
    "Present trails in a structured, readable format with key stats. "
    "Suggest a logical day-by-day plan when designing multi-day trips."
)


@dataclass
class WikilocDeps:
    pass  # Wikiloc client is module-level; no per-request state needed


def _format_trail_detail(t: TrailDetail) -> str:
    lines = [f"**{t.name}**"]
    if t.url:
        lines.append(f"URL: {t.url}")
    if t.author:
        lines.append(f"Author: {t.author}")
    lines.append(f"Activity: {t.activity_type}")
    if t.route_type:
        lines.append(f"Route type: {t.route_type}")
    if t.difficulty:
        lines.append(f"Difficulty: {t.difficulty}")
    if t.distance_km is not None:
        lines.append(f"Distance: {t.distance_km} km")
    if t.elevation_gain_m is not None:
        lines.append(f"Elevation gain: {t.elevation_gain_m} m")
    if t.elevation_loss_m is not None:
        lines.append(f"Elevation loss: {t.elevation_loss_m} m")
    if t.max_altitude_m is not None:
        lines.append(f"Max altitude: {t.max_altitude_m} m")
    if t.estimated_duration:
        lines.append(f"Duration: {t.estimated_duration}")
    lines.append(f"Location: {t.location}")
    if t.country:
        lines.append(f"Country: {t.country}")
    if t.region:
        lines.append(f"Region: {t.region}")
    if t.near_cities:
        lines.append(f"Near: {', '.join(t.near_cities)}")
    if t.best_season:
        lines.append(f"Best season: {t.best_season}")
    if t.surface_type:
        lines.append(f"Surface: {t.surface_type}")
    if t.rating is not None:
        reviews = f" ({t.reviews_count} reviews)" if t.reviews_count else ""
        lines.append(f"Rating: {t.rating}/5{reviews}")
    if t.description:
        snippet = t.description[:800]
        if len(t.description) > 800:
            snippet += "..."
        lines.append(f"\nDescription:\n{snippet}")
    if t.highlights:
        lines.append(f"\nHighlights: {', '.join(t.highlights)}")
    if t.tags:
        lines.append(f"Tags: {', '.join(t.tags)}")
    if t.warnings:
        warning_lines = "\n".join(f"  - {w}" for w in t.warnings)
        lines.append(f"\nWarnings:\n{warning_lines}")
    return "\n".join(lines)


def build_agent() -> Agent:
    """Create and return the Wikiloc trail planning agent with all tools registered."""
    trail_agent: Agent = Agent(
        "openai:gpt-4o-mini",
        system_prompt=SYSTEM_PROMPT,
        deps_type=WikilocDeps,
    )

    @trail_agent.tool
    def search_wikiloc_trails(
        ctx: RunContext[WikilocDeps],
        query: str,
        activity: str = "hiking",
        page: int = 1,
    ) -> str:
        """Search Wikiloc for trails matching a location or keyword.

        Args:
            query: Location or keyword, e.g. "Zakopane", "Tatry ridge", "Dolomites".
            activity: One of: hiking, cycling, mountain-biking, trail-running, running,
                      walking, ski-touring, snowshoeing, via-ferrata, climbing,
                      horse-riding, kayaking. Defaults to hiking.
            page: Results page (1-based). Each page has ~20 trails.

        Returns:
            Formatted list of trails with key stats.
        """
        results: TrailSearchResults = search_trails(query, activity=activity, page=page)
        if not results.trails:
            return f"No trails found for '{query}' ({activity})."

        lines = [f"Found {len(results.trails)} trails for '{query}' ({activity}):"]
        for i, t in enumerate(results.trails, 1):
            parts = [f"{i}. **{t.name}**"]
            if t.difficulty:
                parts.append(f"  Difficulty: {t.difficulty}")
            if t.distance_km is not None:
                parts.append(f"  Distance: {t.distance_km} km")
            if t.elevation_gain_m is not None:
                parts.append(f"  Elevation gain: {t.elevation_gain_m} m")
            if t.estimated_duration:
                parts.append(f"  Duration: {t.estimated_duration}")
            if t.location:
                parts.append(f"  Location: {t.location}")
            if t.rating is not None:
                reviews = f" ({t.reviews_count} reviews)" if t.reviews_count else ""
                parts.append(f"  Rating: {t.rating}/5{reviews}")
            parts.append(f"  URL: {t.url}")
            lines.append("\n".join(parts))

        return "\n\n".join(lines)

    @trail_agent.tool
    def get_wikiloc_trail_details(
        ctx: RunContext[WikilocDeps],
        trail_url: str,
    ) -> str:
        """Fetch full details for a specific Wikiloc trail page.

        Args:
            trail_url: Full URL of the trail, e.g. https://www.wikiloc.com/hiking-trails/...

        Returns:
            Detailed trail information including description, waypoints, warnings, etc.
        """
        detail: TrailDetail | None = get_trail_details(trail_url)
        if detail is None:
            return f"Could not retrieve details for: {trail_url}"
        return _format_trail_detail(detail)

    @trail_agent.tool
    def find_trails_with_full_details(
        ctx: RunContext[WikilocDeps],
        query: str,
        activity: str = "hiking",
        max_trails: int = 3,
    ) -> str:
        """Search Wikiloc and return full details for the top results.

        Use this when the user wants rich recommendations (description, highlights,
        warnings, best season, etc.). Slower than search_wikiloc_trails — keep
        max_trails low (2-4) to avoid long waits.

        Args:
            query: Location or keyword.
            activity: Activity type (default: hiking).
            max_trails: How many trails to enrich with full details (max 5).

        Returns:
            Full details for each trail, formatted for itinerary planning.
        """
        max_trails = max(1, min(max_trails, 5))
        trails = find_trails_with_details(query, activity=activity, max_trails=max_trails)

        if not trails:
            return f"No trails found for '{query}' ({activity})."

        sections = [f"Top {len(trails)} trails for '{query}' ({activity}):"]
        for i, t in enumerate(trails, 1):
            sections.append(f"--- Trail {i} ---")
            sections.append(_format_trail_detail(t))

        return "\n\n".join(sections)

    return trail_agent


def main() -> None:
    """Run an interactive CLI trail planning assistant."""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = input("Please enter your OpenAI API Key: ").strip()

    agent = build_agent()
    deps = WikilocDeps()

    print("Trail Planning Assistant (powered by Wikiloc)")
    print("Ask me to find trails, plan itineraries, or compare routes.")
    print("Ctrl+C to quit.\n")

    try:
        while True:
            query = input("You: ").strip()
            if not query:
                print("Please enter a question.\n")
                continue

            print("\nSearching Wikiloc...\n")
            result = agent.run_sync(query, deps=deps)
            print(f"Assistant: {result.output}\n")
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
