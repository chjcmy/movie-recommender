import os
from qdrant_client import QdrantClient
from django.conf import settings

def get_qdrant_client():
    """
    Returns a QdrantClient instance connected to the configured host.
    """
    host = os.getenv('QDRANT_HOST', 'localhost')
    port = int(os.getenv('QDRANT_PORT', 6333))
    
    return QdrantClient(host=host, port=port)

def create_collection_if_not_exists(collection_name, vector_size):
    """
    Creates a collection if it doesn't exist.
    """
    client = get_qdrant_client()
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)
    
    if not exists:
        from qdrant_client.http import models
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")
