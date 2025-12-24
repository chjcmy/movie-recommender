from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import Movie
from .vector_db import get_qdrant_client
from qdrant_client.http import models as qmodels
import os
import json
import numpy as np
import torch
from django.conf import settings
from .algorithms import SASRec

# Load item map once (global variable for simplicity, in prod use cache)
ITEM_MAP_PATH = os.path.join(settings.BASE_DIR, 'models', 'item_map.json')
ITEM_MAP = {}
if os.path.exists(ITEM_MAP_PATH):
    with open(ITEM_MAP_PATH, 'r') as f:
        ITEM_MAP = json.load(f)
        ITEM_MAP = {int(k): int(v) for k, v in ITEM_MAP.items()}

# Global Model Cache
MODELS = {
    "mf": None,
    "ncf": None,
    "wd": None,
    "sasrec": None
}

def get_or_load_model(model_type, device):
    """
    Load PyTorch model into global cache
    """
    global MODELS
    if MODELS[model_type] is not None:
        return MODELS[model_type]

    num_users = 610 + 100 # Approx buffer or load actual config
    num_items = len(ITEM_MAP) + 1 # +1 for padding
    
    # We need exact dimensions from training. 
    # Hardcoding standard dims for now or saving config would be better.
    # Assuming standard used in commands.
    
    if model_type == 'sasrec':
        from .algorithms import SASRec
        model = SASRec(num_items, max_len=50, embed_dim=64).to(device)
        path = os.path.join(settings.BASE_DIR, 'models', 'sasrec_model_retrained.pth')
        
    elif model_type == 'mf':
        from .algorithms import MF
        # Note: MF constructor needs user/item count. 
        # Ideally we save these params. For now, max(item_map) is safe-ish.
        # But wait, MF user count? We don't use User Embedding for anonymous inference.
        model = MF(num_users=1000, num_items=num_items, embed_dim=32).to(device)
        path = os.path.join(settings.BASE_DIR, 'models', 'mf_model_retrained.pth')
        
    elif model_type == 'ncf':
        from .algorithms import NCF
        model = NCF(num_users=1000, num_items=num_items, embed_dim=32).to(device)
        path = os.path.join(settings.BASE_DIR, 'models', 'ncf_model_retrained.pth')
        
    elif model_type == 'wd':
        from .algorithms import WideAndDeep
        model = WideAndDeep(num_users=1000, num_items=num_items, embed_dim=32).to(device)
        path = os.path.join(settings.BASE_DIR, 'models', 'wide_deep_model.pth')
        # Note: W&D might not have a 'retrained' one if we didn't train it in this session. 
        # Creating a placeholder if file missing?
        if not os.path.exists(path): return None

    if os.path.exists(path):
        # Allow partial load for dimension mismatches (flexible)
        try:
            state_dict = torch.load(path, map_location=device)
            # Filter checks could be added here
            model.load_state_dict(state_dict, strict=False) 
        except Exception as e:
            print(f"Error loading {model_type}: {e}")
            return None
            
        model.eval()
        MODELS[model_type] = model
        return model
    return None

@api_view(['GET'])
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
                
        return Response({"results": response_data})

    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
def recommend_mf(request):
    return _infer_generic(request, 'mf')

@api_view(['POST'])
def recommend_ncf(request):
    return _infer_generic(request, 'ncf')

@api_view(['POST'])
def recommend_wd(request):
    return _infer_generic(request, 'wd')

@api_view(['POST'])
def recommend_sasrec(request):
    return _infer_sasrec(request)

def _infer_sasrec(request):
    """
    True SASRec Inference
    """
    try:
        # In DRF, request.data handles JSON parsing automatically
        data = request.data
        raw_movie_ids = data.get('movie_ids', [])
        limit = int(data.get('limit', 10))
        
        # 1. Prepare Data
        input_indices = [ITEM_MAP[mid] for mid in raw_movie_ids if mid in ITEM_MAP]
        if not input_indices: return Response({"error": "Need valid movies"}, status=400)
        
        # 2. Load Model
        device = torch.device("cpu") # Inference is fast enough on CPU
        model = get_or_load_model('sasrec', device)
        if not model: return Response({"error": "Model not found"}, status=500)
        
        # 3. Predict
        max_len = 50
        seq = [0] * max_len
        seq_len = len(input_indices)
        if seq_len > max_len:
            seq = input_indices[-max_len:]
        else:
            seq[-seq_len:] = input_indices
            
        seq_tensor = torch.LongTensor([seq]).to(device)
        
        with torch.no_grad():
            logits = model(seq_tensor) 
            # Logits shape: (1, 50, NumItems)
            # We want the prediction for the LAST item in the sequence
            last_logits = logits[0, -1, :] # (NumItems,)
            
            # 4. Top K
            # Set scores for input items to -inf to avoid recommending themselves
            for idx in input_indices:
                if idx < len(last_logits):
                    last_logits[idx] = -float('inf')
            
            # Padding index 0 should effectively be ignored or handled
            last_logits[0] = -float('inf')

            top_scores, top_indices = torch.topk(last_logits, k=limit)
            
            # Map back to Movie IDs
            # Be careful: ITEM_MAP is ID -> Index. We need Index -> ID.
            idx2id = {v: k for k, v in ITEM_MAP.items()}
            
            results = []
            for score, idx in zip(top_scores, top_indices):
                idx = int(idx)
                mid = idx2id.get(idx)
                if mid:
                    m = Movie.objects.filter(id=mid).first()
                    results.append({
                        "id": mid,
                        "title": m.title if m else "Unknown",
                        "genres": m.genres if m else "",
                        "score": float(score)
                    })
            
            return Response({"results": results})

    except Exception as e:
        import traceback
        return Response({"error": str(e), "trace": traceback.format_exc()}, status=500)

def _infer_generic(request, model_type):
    """
    Inference for MF/NCF/WD using Item Embedding similarity.
    Since we have no UserID, we find items similar to the history average.
    """
    try:
        data = request.data
        raw_movie_ids = data.get('movie_ids', [])
        limit = int(data.get('limit', 10))
        
        # 1. Prepare Inputs
        input_indices = [ITEM_MAP[mid] for mid in raw_movie_ids if mid in ITEM_MAP]
        if not input_indices: return Response({"error": "Need valid movies"}, status=400)
        
        device = torch.device("cpu")
        model = get_or_load_model(model_type, device)
        if not model: return Response({"error": "Model not found"}, status=500)
        
        with torch.no_grad():
            # Get Item Embeddings
            # MF: item_embedding.weight
            # NCF: gmf_item_embedding.weight (simplified choice)
            # WD: item_embedding.weight
            
            if model_type == 'mf' or model_type == 'wd':
                all_items = model.item_embedding.weight # (NumItems, Dim)
            elif model_type == 'ncf':
                all_items = model.gmf_item_embedding.weight # Use GMF part for simplicity
            
            # Compute History Vector (Mean)
            history_vecs = all_items[input_indices] # (Len, Dim)
            user_vec = torch.mean(history_vecs, dim=0, keepdim=True) # (1, Dim)
            
            # Dot Product Similarity
            # (1, Dim) @ (NumItems, Dim).T -> (1, NumItems)
            scores = torch.matmul(user_vec, all_items.t()).squeeze(0)
            
            # Mask inputs
            for idx in input_indices:
                if idx < len(scores):
                    scores[idx] = -float('inf')
            scores[0] = -float('inf') # Mask padding
            
            top_scores, top_indices = torch.topk(scores, k=limit)
            
            idx2id = {v: k for k, v in ITEM_MAP.items()}
            results = []
            for score, idx in zip(top_scores, top_indices):
                idx = int(idx)
                mid = idx2id.get(idx)
                if mid:
                    m = Movie.objects.filter(id=mid).first()
                    results.append({
                        "id": mid,
                        "title": m.title if m else "Unknown",
                        "genres": m.genres if m else "",
                        "score": float(score)
                    })
                    
            return Response({"results": results})
            
    except Exception as e:
        import traceback
        return Response({"error": str(e), "trace": traceback.format_exc()}, status=500)

@api_view(['GET'])
def search_movies(request):
    """
    GET /api/recommend/search/?query=Toy
    """
    query = request.GET.get('query', '')
    if not query:
        return Response({"results": []})
        
    # Simple case-insensitive contains search
    movies = Movie.objects.filter(title__icontains=query)[:20]
    
    results = []
    for m in movies:
        results.append({
            "id": m.id,
            "title": m.title,
            "genres": m.genres,
        })
        
    return Response({"results": results})
