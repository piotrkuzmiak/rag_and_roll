import os
from typing import List

import google.generativeai as genai
from src.firecrawl_form_scraper import extract_structured_data, TravelDestination
from src.chromadb_load_data import create_update_chromadb_collection
from chromadb.utils import embedding_functions

MAIN_ENTRY_URL = "https://wikivoyage.org"

model =  genai.GenerativeModel("gemini-2.5-flash")

def build_prompt(query: str, context: List[str]) -> str:
    """
    Builds a prompt for the LLM. #

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A prompt for the LLM (str).
    """

    base_prompt = {
        "content": "I am going to ask you a question, which I would like you to answer"
        " based only on the provided context, and not any other information."
        " If there is not enough information in the context to answer the question,"
        ' say "I am not sure", then try to make a guess.'
        " Break your answer up into nicely readable paragraphs.",
    }
    user_prompt = {
        "content": f" The question is '{query}'. Here is all the context you have:"
        f'{(" ").join(context)}',
    }

    # combine the prompts to output a single prompt string
    system = f"{base_prompt['content']} {user_prompt['content']}"

    return system


def get_gemini_response(query: str, context: List[str]) -> str:
    """
    Queries the Gemini API to get a response to the question.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A response to the question.
    """

    response = model.generate_content(build_prompt(query, context))

    return response.text

# def main():
#     print("Hello from rag-and-roll!")
#     destinations = extract_structured_data(MAIN_ENTRY_URL)
#     create_update_chromadb_collection(destinations)

def main(
    collection_name: str = "documents_collection", persist_directory: str = "."
) -> None:
    # Check if the GOOGLE_API_KEY environment variable is set. Prompt the user to set it if not.
    google_api_key = None
    if "GOOGLE_API_KEY" not in os.environ:
        gapikey = input("Please enter your Google API Key: ")
        genai.configure(api_key=gapikey)
        google_api_key = gapikey
    else:
        google_api_key = os.environ["GOOGLE_API_KEY"]

    # create embedding function
    embedding_function = embedding_functions.GoogleGenerativeAIEmbeddingFunction(
        api_key=google_api_key, task_type="RETRIEVAL_QUERY"
    )

    #Fetch data for the collection. In this case, we are fetching travel destination information from Wikivoyage.
    travel_result = extract_structured_data(
        url=MAIN_ENTRY_URL,
        search_term="travel destinations",
        input_selector="#searchInput",
        schema_model=TravelDestination,
        prompt="Extract travel destination information from this webpage. Focus on the name, location, description, best time to visit, attractions, difficulty level, and duration in days for each destination.",
    )

    # Get the collection.
    collection = create_update_chromadb_collection(collection_name=collection_name, travel_result=travel_result)
    # We use a simple input loop.
    while True:
        # Get the user's query
        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print("\nThinking...\n")

        # Query the collection to get the 5 most relevant results
        results = collection.query(
            query_texts=[query], n_results=5, include=["documents", "metadatas"]
        )

        sources = "\n".join(
            [
                f"{result['filename']}: line {result['line_number']}"
                for result in results["metadatas"][0]  # type: ignore
            ]
        )

        # Get the response from Gemini
        response = get_gemini_response(query, results["documents"][0])  # type: ignore

        # Output, with sources
        print(response)
        print("\n")
        print(f"Source documents:\n{sources}")
        print("\n")

if __name__ == "__main__":
    main()
