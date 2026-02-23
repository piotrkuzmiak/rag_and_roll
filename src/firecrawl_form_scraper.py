from firecrawl import Firecrawl
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Firecrawl
app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

def scrape_search_results(url: str, search_term: str, input_selector: str):
    """
    Scrape search results by injecting text into a search form.

    Args:
        url: The URL of the page with the search form
        search_term: The text to inject into the search input
        input_selector: CSS selector for the search input field
    """
    # Define actions to perform on the page
    actions = [
        {
            "type": "write",
            "selector": input_selector,
            "text": search_term
        },
        {
            "type": "press",
            "key": "Enter"
        }
    ]

    # Scrape the page with actions
    result = app.scrape(
        url,
        formats=['markdown'],  # You can change this to 'html' or other formats
        actions=actions,
        only_main_content=True,  # Focus on main content only
        timeout=12000,  # Limit processing time (12 seconds)
        remove_base64_images=True,  # Remove images to reduce size
        include_tags=['h1', 'h2', 'h3', 'p', 'li', 'a'],  # Only include specific tags
        exclude_tags=['script', 'style', 'nav', 'footer', 'aside', 'video'],  # Exclude unwanted tags
        max_age=3600,  # Use cached results if less than 1 hour old,
        block_ads=True  # Block ads to focus on relevant content
    )

    return result

def generate_llms_text(results):
    """
    Generate formatted text input for LLM from scraped results.

    Args:
        results: The Document object returned by scrape_search_results

    Returns:
        str: Formatted text suitable for LLM input
    """
    # Extract relevant information from the Document object
    content = results.markdown
    title = results.metadata.title if results.metadata and results.metadata.title else "No title"
    url = results.metadata.url if results.metadata and results.metadata.url else "No URL"
    description = results.metadata.description if results.metadata and results.metadata.description else "No description"

    # Format the text for LLM input
    llm_text = f"""Title: {title}
URL: {url}
Description: {description}

Content:
{content}
"""

    return llm_text

# Example usage
if __name__ == "__main__":
    # Example: Search on Wikiloc
    url = "https://pl.wikiloc.com/"
    search_term = "Tatry"
    input_selector = "input.search-box__input"  # Wikiloc's search input

    results = scrape_search_results(url, search_term, input_selector)
    print("Scraped content:")
    print(results.markdown)

    # Generate LLM input text
    llm_input = generate_llms_text(results)
    print("\nLLM Input Text:")
    print(llm_input)