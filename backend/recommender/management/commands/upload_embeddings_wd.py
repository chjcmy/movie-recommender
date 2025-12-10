import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from django.core.management.base import BaseCommand
from django.conf import settings
from recommender.algorithms import WideAndDeep
from recommender.vector_db import get_qdrant_client, create_collection_if_not_exists
from qdrant_client.http import models as qmodels
from qdrant_client import QdrantClient

class Command(BaseCommand):
    help = 'Train Wide & Deep model and upload embeddings to Qdrant'

    def handle(self, *args, **kwargs):
        # 1. Load Data & Mapping
        data_dir = os.path.join(settings.BASE_DIR, 'data')
        ratings_path = os.path.join(data_dir, 'ratings.csv')
        movies_path = os.path.join(data_dir, 'movies.csv')
        mapping_path = os.path.join(settings.BASE_DIR, 'models', 'item_map.json')
        
        if not os.path.exists(mapping_path):
            self.stdout.write(self.style.ERROR("Error: item_map.json not found. Run upload_embeddings first!"))
            return

        self.stdout.write("1. Loading Data & Mapping...")
        with open(mapping_path, 'r') as f:
            item2idx = json.load(f)
            item2idx = {int(k): int(v) for k, v in item2idx.items()}
            
        idx2item = {v: k for k, v in item2idx.items()}
        num_items = len(item2idx)
        
        # Load Ratings
        df = pd.read_csv(ratings_path)
        user_ids = df['userId'].unique()
        user2idx = {u: i for i, u in enumerate(user_ids)}
        num_users = len(user_ids)
        
        df['user_idx'] = df['userId'].map(user2idx)
        df['item_idx'] = df['movieId'].map(item2idx)
        
        # Filter out unknown items
        df = df.dropna(subset=['item_idx'])
        df['item_idx'] = df['item_idx'].astype(int)
        
        self.stdout.write(f"   Users: {num_users}, Items: {num_items}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        self.stdout.write(f"   Using device: {device}")

        # 2. Wide & Deep Training/Loading
        self.stdout.write("\n2. Processing Wide & Deep...")
        model_wd_path = os.path.join(settings.BASE_DIR, 'models', 'wd_model_retrained.pth')
        
        # Need Genres
        movies_df = pd.read_csv(movies_path)
        genres_set = set()
        for g in movies_df['genres']:
            genres_set.update(g.split('|'))
        genres_list = sorted(list(genres_set))
        genre2idx = {g: i for i, g in enumerate(genres_list)}
        num_genres = len(genres_list)
        
        model_wd = WideAndDeep(num_users, num_items, num_genres, embed_dim=32).to(device)
        
        if os.path.exists(model_wd_path):
            self.stdout.write(f"   Loading existing W&D model from {model_wd_path}...")
            model_wd.load_state_dict(torch.load(model_wd_path, map_location=device))
        else:
            optimizer = torch.optim.Adam(model_wd.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            item_genre_map = {}
            for _, row in movies_df.iterrows():
                mid = row['movieId']
                if mid in item2idx:
                    idx = item2idx[mid]
                    g_indices = [genre2idx[g] for g in row['genres'].split('|')]
                    vec = np.zeros(num_genres)
                    vec[g_indices] = 1
                    item_genre_map[idx] = vec
            
            users = torch.LongTensor(df['user_idx'].values).to(device)
            items = torch.LongTensor(df['item_idx'].values).to(device)
            ratings = torch.FloatTensor(df['rating'].values).to(device)
            
            self.stdout.write("   Training Wide & Deep (1 Epoch)...")
            model_wd.train()
            
            indices = np.arange(len(df))
            np.random.shuffle(indices)
            batch_size = 10000
            
            for i in range(0, len(df), batch_size):
                batch_idx = indices[i:i+batch_size]
                u_batch = users[batch_idx]
                i_batch = items[batch_idx]
                r_batch = ratings[batch_idx]
                
                g_batch_list = [item_genre_map.get(itm.item(), np.zeros(num_genres)) for itm in i_batch]
                g_batch = torch.FloatTensor(np.array(g_batch_list)).to(device)
                
                optimizer.zero_grad()
                preds = model_wd(u_batch, i_batch, g_batch)
                loss = criterion(preds, r_batch)
                loss.backward()
                optimizer.step()
            
            torch.save(model_wd.state_dict(), model_wd_path)
            self.stdout.write(f"   Model saved to {model_wd_path}")

        # 3. Upload to Qdrant
        collection_name = "movies_wide_deep"
        self.stdout.write(f"\n3. Uploading to {collection_name}...")
        create_collection_if_not_exists(collection_name, 32)
        
        client = QdrantClient(
            host=os.getenv('QDRANT_HOST'), 
            port=int(os.getenv('QDRANT_PORT')), 
            timeout=300
        )
        
        embeddings_wd = model_wd.item_embedding.weight.detach().cpu().numpy()
        
        points = []
        batch_size_upload = 20
        total_uploaded = 0
        
        for idx, embedding in enumerate(embeddings_wd):
            movie_id = idx2item.get(idx)
            if movie_id is None: continue
            
            points.append(qmodels.PointStruct(
                id=int(movie_id),
                vector=embedding.tolist(),
                payload={"movie_id": int(movie_id)}
            ))
            
            if len(points) >= batch_size_upload:
                try:
                    client.upsert(collection_name=collection_name, points=points)
                    total_uploaded += len(points)
                    points = []
                    if total_uploaded % 1000 == 0:
                        self.stdout.write(f"   Uploaded {total_uploaded} vectors...")
                except Exception as e:
                    self.stdout.write(f"Error: {e}")
        
        if points:
            client.upsert(collection_name=collection_name, points=points)
        self.stdout.write(f"   Done! {total_uploaded} vectors uploaded to {collection_name}.")
