import os
import json
import torch
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import MultiLabelBinarizer
from .algorithms import MF, NCF, WideAndDeep, SASRec

# --- Global Variables for Models and Data ---
# In a production setting, these should be handled more robustly (e.g., cache, separate service)
MODELS = {}
MAPPINGS = {}
DATA = {}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_data_and_models():
    """
    Load data to rebuild mappings and load trained models.
    """
    global MODELS, MAPPINGS, DATA
    
    if MODELS and MAPPINGS:
        return

    print("Loading data and rebuilding mappings...")
    
    try:
        ratings = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
        movies = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
    except FileNotFoundError:
        print("Data files not found. Skipping model loading.")
        return
    
    # --- 1. MF / NCF Mappings (from ratings.csv directly) ---
    # Logic: df = pd.read_csv('../data/ratings.csv'); df = df.sample(frac=0.1, random_state=42)
    print("Building MF/NCF mappings...")
    mf_data = ratings.sample(frac=0.1, random_state=42)
    
    mf_user_ids = mf_data['userId'].unique()
    mf_item_ids = mf_data['movieId'].unique()
    
    mf_user2idx = {u: i for i, u in enumerate(mf_user_ids)}
    mf_item2idx = {m: i for i, m in enumerate(mf_item_ids)}
    mf_idx2item = {i: m for m, i in mf_item2idx.items()}
    
    MAPPINGS['mf'] = {
        'user2idx': mf_user2idx,
        'item2idx': mf_item2idx,
        'idx2item': mf_idx2item
    }
    
    # --- 2. Wide & Deep Mappings (from merged data) ---
    # Logic: data = pd.merge(ratings, movies, on='movieId'); data = data.sample(frac=0.1, random_state=42)
    print("Building Wide & Deep mappings...")
    wd_data = pd.merge(ratings, movies, on='movieId')
    wd_data = wd_data.sample(frac=0.1, random_state=42)
    
    wd_user_ids = wd_data['userId'].unique()
    wd_item_ids = wd_data['movieId'].unique()
    
    wd_user2idx = {u: i for i, u in enumerate(wd_user_ids)}
    wd_item2idx = {m: i for i, m in enumerate(wd_item_ids)}
    wd_idx2item = {i: m for m, i in wd_item2idx.items()}
    
    # Genres
    wd_data['genres_list'] = wd_data['genres'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    mlb.fit(wd_data['genres_list'])
    
    movie_genre_map = {}
    sample_movies = wd_data[['movieId', 'genres_list']].drop_duplicates('movieId')
    sample_movies_encoded = mlb.transform(sample_movies['genres_list'])
    for mid, vec in zip(sample_movies['movieId'], sample_movies_encoded):
        movie_genre_map[mid] = vec
        
    MAPPINGS['wide_deep'] = {
        'user2idx': wd_user2idx,
        'item2idx': wd_item2idx,
        'idx2item': wd_idx2item,
        'mlb': mlb,
        'movie_genre_map': movie_genre_map
    }

    # --- 3. Load Models ---
    print("Loading models...")

    # MF
    try:
        mf_model = MF(len(mf_user_ids), len(mf_item_ids))
        try:
            mf_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'mf_model.pth'), map_location=DEVICE))
        except Exception as e:
            print(f"Warning: Failed to load MF weights ({e}). Using random weights.")
        mf_model.to(DEVICE)
        mf_model.eval()
        MODELS['mf'] = mf_model
    except Exception as e:
        print(f"Failed to initialize MF model: {e}")

    # NCF
    try:
        ncf_model = NCF(len(mf_user_ids), len(mf_item_ids))
        try:
            ncf_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'ncf_model.pth'), map_location=DEVICE))
        except Exception as e:
            print(f"Warning: Failed to load NCF weights ({e}). Using random weights.")
        ncf_model.to(DEVICE)
        ncf_model.eval()
        MODELS['ncf'] = ncf_model
    except Exception as e:
        print(f"Failed to initialize NCF model: {e}")

    # Wide & Deep
    try:
        wd_model = WideAndDeep(len(wd_user_ids), len(wd_item_ids), len(mlb.classes_))
        try:
            wd_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'wide_deep_model.pth'), map_location=DEVICE))
        except Exception as e:
            print(f"Warning: Failed to load Wide & Deep weights ({e}). Using random weights.")
        wd_model.to(DEVICE)
        wd_model.eval()
        MODELS['wide_deep'] = wd_model
    except Exception as e:
        print(f"Failed to initialize Wide & Deep model: {e}")
    # SASRec
    try:
        # SASRec uses 1-based indexing for items (0 is padding)
        # So num_embeddings should be num_items + 1
        sasrec_model = SASRec(len(mf_item_ids) + 1, 50) # max_len=50
        try:
            sasrec_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'sasrec_model.pth'), map_location=DEVICE))
        except Exception as e:
            print(f"Warning: Failed to load SASRec weights ({e}). Using random weights.")
        sasrec_model.to(DEVICE)
        sasrec_model.eval()
        MODELS['sasrec'] = sasrec_model
    except Exception as e:
        print(f"Failed to initialize SASRec model: {e}")
        
    print("Models loaded.")

# Initialize on module import (or first request)
load_data_and_models()

def get_top_k(model, user_idx, all_item_indices, k=10, genres=None):
    """Helper to get top K recommendations"""
    user_tensor = torch.LongTensor([user_idx] * len(all_item_indices)).to(DEVICE)
    item_tensor = torch.LongTensor(all_item_indices).to(DEVICE)
    
    with torch.no_grad():
        if genres is not None:
            # Wide & Deep
            genres_tensor = torch.FloatTensor(genres).to(DEVICE)
            predictions = model(user_tensor, item_tensor, genres_tensor)
        else:
            # MF / NCF
            predictions = model(user_tensor, item_tensor)
            
    # Get top K indices
    _, top_indices = torch.topk(predictions, k)
    return top_indices.cpu().numpy()

def recommend_mf(request, user_id):
    if 'mf' not in MODELS:
        return JsonResponse({'error': 'MF Model not loaded'}, status=503)
    
    mapping = MAPPINGS['mf']
    user2idx = mapping['user2idx']
    idx2item = mapping['idx2item']
    
    if user_id not in user2idx:
        return JsonResponse({'error': 'User not found in training set'}, status=404)
        
    user_idx = user2idx[user_id]
    all_items = list(mapping['item2idx'].values())
    
    top_indices = get_top_k(MODELS['mf'], user_idx, all_items)
    recommendations = [int(idx2item[i]) for i in top_indices]
    
    return JsonResponse({'user_id': user_id, 'recommendations': recommendations})

def recommend_ncf(request, user_id):
    if 'ncf' not in MODELS:
        return JsonResponse({'error': 'NCF Model not loaded'}, status=503)
        
    mapping = MAPPINGS['mf'] # NCF uses same mapping as MF
    user2idx = mapping['user2idx']
    idx2item = mapping['idx2item']
    
    if user_id not in user2idx:
        return JsonResponse({'error': 'User not found in training set'}, status=404)
        
    user_idx = user2idx[user_id]
    all_items = list(mapping['item2idx'].values())
    
    top_indices = get_top_k(MODELS['ncf'], user_idx, all_items)
    recommendations = [int(idx2item[i]) for i in top_indices]
    
    return JsonResponse({'user_id': user_id, 'recommendations': recommendations})

def recommend_wide_deep(request, user_id):
    if 'wide_deep' not in MODELS:
        return JsonResponse({'error': 'Wide & Deep Model not loaded'}, status=503)
        
    mapping = MAPPINGS['wide_deep']
    user2idx = mapping['user2idx']
    idx2item = mapping['idx2item']
    movie_genre_map = mapping['movie_genre_map']
    
    if user_id not in user2idx:
        return JsonResponse({'error': 'User not found in training set'}, status=404)
        
    user_idx = user2idx[user_id]
    all_items = list(mapping['item2idx'].values())
    
    valid_items = []
    valid_genres = []
    
    for i in all_items:
        mid = idx2item[i]
        if mid in movie_genre_map:
            valid_items.append(i)
            valid_genres.append(movie_genre_map[mid])
            
    if not valid_items:
        return JsonResponse({'error': 'No valid items found'}, status=500)
        
    top_indices = get_top_k(MODELS['wide_deep'], user_idx, valid_items, genres=valid_genres)
    recommendations = [int(idx2item[valid_items[i]]) for i in top_indices]
    
    return JsonResponse({'user_id': user_id, 'recommendations': recommendations})

@csrf_exempt
def recommend_sasrec(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=405)
    
    if 'sasrec' not in MODELS:
        return JsonResponse({'error': 'SASRec Model not loaded'}, status=503)
        
    try:
        data = json.loads(request.body)
        movie_ids = data.get('movie_ids', [])
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
        
    if not movie_ids:
        return JsonResponse({'error': 'No movie_ids provided'}, status=400)
        
    # Use MF mappings for item indices (assuming consistent item set)
    mapping = MAPPINGS.get('mf')
    if not mapping:
         return JsonResponse({'error': 'Mappings not loaded'}, status=503)
         
    item2idx = mapping['item2idx']
    idx2item = mapping['idx2item']
    
    # Convert Movie IDs to Indices
    # SASRec expects 1-based indexing (0 is padding)
    # Our item2idx is 0-based. So we add 1.
    input_seq = []
    for mid in movie_ids:
        if mid in item2idx:
            input_seq.append(item2idx[mid] + 1)
    
    if not input_seq:
        return JsonResponse({'error': 'None of the provided movies are in the training set'}, status=404)
        
    # Pad/Truncate
    max_len = 50
    if len(input_seq) < max_len:
        input_seq = [0] * (max_len - len(input_seq)) + input_seq
    else:
        input_seq = input_seq[-max_len:]
        
    input_tensor = torch.LongTensor([input_seq]).to(DEVICE)
    
    with torch.no_grad():
        logits = MODELS['sasrec'](input_tensor)
        # Last time step prediction
        last_logits = logits[0, -1, :] 
        
        # Get Top 10
        # We should ignore index 0 (padding) if it comes up
        # last_logits[0] = -float('inf') 
        
        _, top_indices = torch.topk(last_logits, 10)
        
    # Convert back to Movie IDs
    # Index i corresponds to item2idx value i-1
    recommendations = []
    for i in top_indices.cpu().numpy():
        if i == 0: continue
        original_idx = i - 1
        if original_idx in idx2item:
            recommendations.append(int(idx2item[original_idx]))
            
    return JsonResponse({'input_movies': movie_ids, 'recommendations': recommendations})
