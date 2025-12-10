from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from .models import Movie
from .vector_db import get_qdrant_client
from qdrant_client.http import models as qmodels
import os

def find_similar_movies(request, movie_id):
    try:
        limit = int(request.GET.get('limit', 10))
        collection_name = request.GET.get('model', 'movies_mf') # Default to MF
        
        client = get_qdrant_client()
        
        search_result = client.recommend(
            collection_name=collection_name,
            positive=[int(movie_id)],
            limit=limit
        )
        
        similar_movie_ids = [point.id for point in search_result]
        
        movies = Movie.objects.filter(id__in=similar_movie_ids)
        movie_map = {m.id: m for m in movies}
        
        response_data = []
        for point in search_result:
            mid = point.id
            movie = movie_map.get(mid)
            if movie:
                response_data.append({
                    "id": movie.id,
                    "title": movie.title,
                    "genres": movie.genres,
                    "score": point.score,
                    "poster_url": ""
                })
                
        return JsonResponse({"results": response_data})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
