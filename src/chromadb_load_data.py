import chromadb

def create_update_chromadb_collection(data, collection_name: str = "travel_destinations") -> chromadb.Collection:
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name)
    collection.add(
        ids=[data.pop("name")],
        documents=[data.pop("description")],
        metadatas=[data]
    )
    return collection