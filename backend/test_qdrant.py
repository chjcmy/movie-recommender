from recommender.vector_db import get_qdrant_client
import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

try:
    client = get_qdrant_client()
    collections = client.get_collections()
    print(f"Successfully connected to Qdrant!")
    print(f"Collections: {collections}")
except Exception as e:
    print(f"Failed to connect to Qdrant: {e}")
