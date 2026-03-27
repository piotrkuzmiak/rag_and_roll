"""
Wikiloc scraper for trekking routes and outdoor activities.
Designed as a tool set for AI travel assistant agents.

Usage:
    from src.wikiloc_scraper import search_trails, get_trail_details, find_trails_with_details

    # Search for hiking trails near Zakopane
    results = search_trails("Zakopane", activity="hiking")

    # Get full details for a specific trail
    detail = get_trail_details("https://www.wikiloc.com/hiking-trails/...")

    # Search and automatically enrich with details
    trails = find_trails_with_details("Tatry", activity="hiking", max_trails=3)
"""

import os
from typing import Optional

from dotenv import load_dotenv
from firecrawl import Firecrawl
from firecrawl.types import (
    ExecuteJavascriptAction,
    JsonFormat,
    WaitAction,
)
from pydantic import BaseModel, Field

load_dotenv()

# HTTP client timeout in seconds — must be > server-side timeout in scrape options.
# Wikiloc is JS-heavy and may need 60+ seconds with cookie dismissal + LLM extraction.
client = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"), timeout=120)

# Wikiloc activity type IDs (from their URL/API)
ACTIVITY_IDS: dict[str, int] = {
    "hiking": 1,
    "cycling": 2,
    "mountain-biking": 3,
    "trail-running": 23,
    "running": 14,
    "walking": 57,
    "ski-touring": 15,
    "snowshoeing": 37,
    "via-ferrata": 28,
    "climbing": 26,
    "horse-riding": 10,
    "kayaking": 9,
}

# JavaScript to dismiss cookie/GDPR consent popups
_DISMISS_CONSENT_JS = """
(function() {
    const selectors = [
        '#didomi-notice-agree-button',
        '.didomi-continue-without-agreeing',
        '[data-testid="cookie-accept"]',
        '.fc-cta-consent',
        '#onetrust-accept-btn-handler',
    ];
    for (const sel of selectors) {
        const el = document.querySelector(sel);
        if (el) { el.click(); break; }
    }
})();
"""


# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------

class TrailSummary(BaseModel):
    """Summary of a trail from search results — lightweight, enough for a list view."""

    name: str = Field(description="Name of the trail")
    url: str = Field(description="Full URL to the trail page on wikiloc.com")
    activity_type: str = Field(description="Activity type, e.g. hiking, cycling, trail-running")
    difficulty: Optional[str] = Field(
        None, description="Difficulty level: Easy, Moderate, Hard, or Expert"
    )
    distance_km: Optional[float] = Field(None, description="Total trail distance in kilometres")
    elevation_gain_m: Optional[int] = Field(None, description="Total elevation gain in metres")
    estimated_duration: Optional[str] = Field(
        None, description="Human-readable estimated completion time, e.g. '3h 20min'"
    )
    location: Optional[str] = Field(
        None, description="Location name or area, e.g. 'Zakopane, Poland'"
    )
    rating: Optional[float] = Field(None, description="Average user rating from 1 to 5")
    reviews_count: Optional[int] = Field(None, description="Number of user reviews/comments")


class TrailSearchResults(BaseModel):
    """Container for search results from a Wikiloc search page."""

    trails: list[TrailSummary] = Field(
        default_factory=list,
        description="List of trails found — order matches Wikiloc's relevance ranking",
    )
    total_results_hint: Optional[int] = Field(
        None, description="Total number of results Wikiloc reports for this query, if shown"
    )


class TrailDetail(BaseModel):
    """Full details about a single Wikiloc trail — everything an AI agent needs to reason about."""

    # Identity
    name: str = Field(description="Full trail name as shown on Wikiloc")
    url: Optional[str] = Field(None, description="Canonical URL of the trail page")
    author: Optional[str] = Field(None, description="Wikiloc username who uploaded the trail")

    # Activity
    activity_type: str = Field(description="Primary activity type, e.g. hiking")
    route_type: Optional[str] = Field(
        None, description="Route shape: Loop, One-way, or Out-and-back"
    )
    difficulty: Optional[str] = Field(
        None, description="Difficulty: Easy, Moderate, Hard, or Expert"
    )

    # Key stats
    distance_km: Optional[float] = Field(None, description="Total distance in kilometres")
    elevation_gain_m: Optional[int] = Field(None, description="Total elevation gain in metres")
    elevation_loss_m: Optional[int] = Field(None, description="Total elevation loss in metres")
    max_altitude_m: Optional[int] = Field(None, description="Highest point on the trail in metres")
    min_altitude_m: Optional[int] = Field(None, description="Lowest point on the trail in metres")
    estimated_duration: Optional[str] = Field(
        None, description="Estimated completion time, e.g. '4h 30min'"
    )

    # Location
    location: str = Field(description="Specific location or area name")
    country: Optional[str] = Field(None, description="Country where the trail is located")
    region: Optional[str] = Field(None, description="Region, state, or voivodeship")
    near_cities: Optional[list[str]] = Field(
        None, description="Nearest cities or towns useful for trip planning"
    )

    # Content
    description: Optional[str] = Field(
        None,
        description="Full trail description including access info, terrain notes, and highlights",
    )
    highlights: Optional[list[str]] = Field(
        None, description="Notable points of interest or scenic spots along the route"
    )
    tags: Optional[list[str]] = Field(
        None, description="Wikiloc category tags, e.g. ['mountains', 'forest', 'waterfall']"
    )
    surface_type: Optional[str] = Field(
        None, description="Dominant trail surface: dirt, rocky, paved, mixed, snow"
    )
    best_season: Optional[str] = Field(
        None, description="Recommended months or season to visit, e.g. 'June to September'"
    )
    warnings: Optional[list[str]] = Field(
        None, description="Any safety notes, seasonal closures, or hazards mentioned"
    )

    # Social proof
    rating: Optional[float] = Field(None, description="Average user rating (1–5)")
    reviews_count: Optional[int] = Field(None, description="Total number of user reviews")
    photos_count: Optional[int] = Field(None, description="Number of photos uploaded by users")
    waypoints_count: Optional[int] = Field(
        None, description="Number of named waypoints or POIs on the route"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

ACTIVITY_SLUGS: dict[str, str] = {
    "hiking": "hiking",
    "cycling": "cycling",
    "mountain-biking": "mountain-biking",
    "trail-running": "trail-running",
    "running": "running",
    "walking": "walking",
    "ski-touring": "ski-touring",
    "snowshoeing": "snowshoeing",
    "via-ferrata": "via-ferrata",
    "climbing": "climbing",
    "horse-riding": "horse-riding",
    "kayaking": "kayaking",
}


# Common location name → Wikiloc URL path segment mappings.
# Wikiloc uses lowercase slugs matching the country/region path in their URL.
LOCATION_PATH_MAP: dict[str, str] = {
    # Countries
    "poland": "poland",
    "france": "france",
    "italy": "italy",
    "spain": "spain",
    "germany": "germany",
    "austria": "austria",
    "switzerland": "switzerland",
    "slovakia": "slovakia",
    "czech republic": "czech-republic",
    "czechia": "czech-republic",
    "norway": "norway",
    "sweden": "sweden",
    "croatia": "croatia",
    "slovenia": "slovenia",
    "portugal": "portugal",
    "greece": "greece",
    "united states": "united-states",
    "usa": "united-states",
    # Polish regions / popular destinations
    "tatry": "poland/lesser-poland",
    "tatra": "poland/lesser-poland",
    "zakopane": "poland/lesser-poland",
    "małopolska": "poland/lesser-poland",
    "lesser poland": "poland/lesser-poland",
    "sudety": "poland/lower-silesian",
    "sudeten": "poland/lower-silesian",
    "bieszczady": "poland/subcarpathian",
    "karkonosze": "poland/lower-silesian",
    # International popular areas
    "dolomites": "italy/trentino-south-tyrol",
    "dolomiti": "italy/trentino-south-tyrol",
    "chamonix": "france/auvergne-rhone-alpes",
    "mont blanc": "france/auvergne-rhone-alpes",
    "alps": "switzerland",
    "pyrenees": "france/occitanie",
    "pireneje": "spain/aragon",
}


def _location_to_path(query: str) -> str:
    """Map a query string to a Wikiloc location path segment, or empty string if unknown."""
    q = query.lower().strip()
    # Direct match
    if q in LOCATION_PATH_MAP:
        return LOCATION_PATH_MAP[q]
    # Partial match — check if any key appears in the query
    for key, path in LOCATION_PATH_MAP.items():
        if key in q:
            return path
    return ""


def _build_search_url(activity: str, location_path: str = "") -> str:
    slug = ACTIVITY_SLUGS.get(activity.lower(), "hiking")
    if location_path:
        return f"https://www.wikiloc.com/trails/{slug}/{location_path}"
    return f"https://www.wikiloc.com/trails/{slug}"


def _consent_actions() -> list:
    """Return Firecrawl actions that dismiss cookie/GDPR banners only."""
    return [
        ExecuteJavascriptAction(script=_DISMISS_CONSENT_JS),
        WaitAction(milliseconds=600),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_trails(
    query: str,
    activity: str = "hiking",
    page: int = 1,
) -> TrailSearchResults:
    """Search Wikiloc for trails matching a keyword or location.

    Args:
        query: Free-text search, e.g. "Zakopane", "Tatry ridge", "Val d'Aran".
        activity: Activity type. Supported values: hiking, cycling, mountain-biking,
                  trail-running, running, walking, ski-touring, snowshoeing,
                  via-ferrata, climbing, horse-riding, kayaking.
                  Defaults to "hiking".
        page: Results page number (1-based). Each page returns ~20 trails.

    Returns:
        TrailSearchResults with a list of TrailSummary objects.
    """
    location_path = _location_to_path(query)
    url = _build_search_url(activity, location_path)
    print(f"[wikiloc] Search URL: {url} | query: '{query}' | location_path: '{location_path}'")
    print("[wikiloc] Scraping search results (this may take 30-60s)...")

    result = client.scrape(
        url,
        actions=_consent_actions(),
        formats=[
            JsonFormat(
                type="json",
                schema=TrailSearchResults.model_json_schema(),
                prompt=(
                    f"Extract trails from this Wikiloc page that are near or related to '{query}'. "
                    "For each trail listed, extract: "
                    "name (trail title), "
                    "url (the full href of the trail link — must start with https://www.wikiloc.com), "
                    "activity_type (hiking/cycling/etc.), "
                    "difficulty (Easy/Moderate/Hard/Expert if shown), "
                    "distance_km (numeric kilometres), "
                    "elevation_gain_m (positive ascent in metres), "
                    "estimated_duration (e.g. '3h 20min'), "
                    "location (place name or region shown on the card), "
                    "rating (numeric 1-5 if shown), "
                    "reviews_count (integer count of reviews/comments). "
                    "Also extract total_results_hint if the page shows a total count of results."
                ),
            )
        ],
        only_main_content=False,
        wait_for=4000,
        timeout=90000,
        remove_base64_images=True,
        block_ads=True,
        proxy="stealth",
    )

    print(f"[wikiloc] Search done. Raw JSON: {result.json}")
    if result.json:
        return TrailSearchResults.model_validate(result.json)
    return TrailSearchResults(trails=[], total_results_hint=None)


def get_trail_details(trail_url: str) -> Optional[TrailDetail]:
    """Fetch full details for a single Wikiloc trail page.

    Args:
        trail_url: Full URL of the trail page, e.g.
                   "https://www.wikiloc.com/hiking-trails/trail-name-12345678"

    Returns:
        TrailDetail with all available information, or None on failure.
    """
    print(f"[wikiloc] Fetching trail details: {trail_url}")
    print("[wikiloc] Scraping trail page (this may take 30-60s)...")
    result = client.scrape(
        trail_url,
        actions=_consent_actions(),
        formats=[
            JsonFormat(
                type="json",
                schema=TrailDetail.model_json_schema(),
                prompt=(
                    "Extract complete trail information from this Wikiloc trail page. Include: "
                    "name (full trail title), "
                    "url (canonical page URL), "
                    "author (username who uploaded), "
                    "activity_type (e.g. hiking), "
                    "route_type (Loop / One-way / Out-and-back), "
                    "difficulty (Easy/Moderate/Hard/Expert), "
                    "distance_km (kilometres as decimal), "
                    "elevation_gain_m (positive ascent metres), "
                    "elevation_loss_m (descent metres), "
                    "max_altitude_m (highest point metres), "
                    "min_altitude_m (lowest point metres), "
                    "estimated_duration (human-readable time), "
                    "location (specific place or area name), "
                    "country, region, "
                    "near_cities (list of closest towns/cities useful for logistics), "
                    "description (full trail description text — preserve detail), "
                    "highlights (list of named viewpoints, peaks, refuges, or POIs), "
                    "tags (Wikiloc category tags), "
                    "surface_type (dominant surface), "
                    "best_season (recommended visiting period), "
                    "warnings (safety notes, seasonal closures, or hazards), "
                    "rating (numeric 1-5), "
                    "reviews_count, photos_count, waypoints_count."
                ),
            )
        ],
        only_main_content=False,
        wait_for=3000,
        timeout=90000,
        remove_base64_images=True,
        block_ads=True,
        proxy="stealth",
    )

    print(f"[wikiloc] Trail detail done. Raw JSON: {result.json}")
    if result.json:
        return TrailDetail.model_validate(result.json)
    return None


def find_trails_with_details(
    query: str,
    activity: str = "hiking",
    max_trails: int = 5,
) -> list[TrailDetail]:
    """Search for trails and enrich each result with full details.

    Convenience function that combines search_trails + get_trail_details.
    Useful when the AI agent needs rich context for a set of recommendations.

    Args:
        query: Search keyword or location name.
        activity: Activity type (default: "hiking").
        max_trails: Maximum number of trails to enrich with details. Keep low
                    to reduce API calls — each trail is one additional scrape.

    Returns:
        List of TrailDetail objects (may be shorter than max_trails if some
        detail pages fail to scrape).
    """
    search_results = search_trails(query, activity)
    trails_to_fetch = search_results.trails[:max_trails]

    details: list[TrailDetail] = []
    for summary in trails_to_fetch:
        if not summary.url or not summary.url.startswith("https://"):
            continue
        detail = get_trail_details(summary.url)
        if detail:
            # Backfill URL from summary if the detail page didn't capture it
            if not detail.url:
                detail = detail.model_copy(update={"url": summary.url})
            details.append(detail)

    return details


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=== Wikiloc Scraper — Demo ===\n")

    # 1. Search for trails
    query = "Zakopane"
    activity = "hiking"
    print(f"Searching for '{query}' ({activity}) trails...\n")

    search_results = search_trails(query, activity)
    print(f"Found {len(search_results.trails)} trails in search results:")
    for i, t in enumerate(search_results.trails, 1):
        print(
            f"  {i}. {t.name} | {t.difficulty or '?'} | "
            f"{t.distance_km or '?'} km | {t.location or '?'}"
        )

    print()

    # 2. Get details for the first result
    if search_results.trails:
        first = search_results.trails[0]
        print(f"Fetching details for: {first.name}\n  URL: {first.url}\n")
        detail = get_trail_details(first.url)
        if detail:
            print("Trail details:")
            print(json.dumps(detail.model_dump(exclude_none=True), indent=2, ensure_ascii=False))
        else:
            print("Failed to fetch trail details.")
