import os
import csv
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from django.core.management.base import BaseCommand
from django.conf import settings
from recommender.algorithms import MF
from recommender.vector_db import get_qdrant_client, create_collection_if_not_exists
from qdrant_client.http import models as qmodels
from qdrant_client import QdrantClient

class Command(BaseCommand):
    help = 'Train MF model and upload embeddings to Qdrant'

    def handle(self, *args, **kwargs):
        data_dir = os.path.join(settings.BASE_DIR, 'data')
        ratings_path = os.path.join(data_dir, 'ratings.csv')
        model_path = os.path.join(settings.BASE_DIR, 'models', 'mf_model_retrained.pth')
        mapping_path = os.path.join(settings.BASE_DIR, 'models', 'item_map.json')
        
        self.stdout.write("1. Loading Data...")

        df = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating'])
        
        user_ids = df['userId'].unique()
        item_ids = df['movieId'].unique()
        
        user2idx = {u: i for i, u in enumerate(user_ids)}
        item2idx = {i: idx for idx, i in enumerate(item_ids)}
        idx2item = {idx: i for i, idx in item2idx.items()}
        
        df['user_idx'] = df['userId'].map(user2idx)
        df['item_idx'] = df['movieId'].map(item2idx)
        
        num_users = len(user_ids)
        num_items = len(item_ids)
        
        self.stdout.write(f"   Users: {num_users}, Items: {num_items}")
        
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        with open(mapping_path, 'w') as f:
            json.dump({str(k): int(v) for k, v in item2idx.items()}, f)
        self.stdout.write(f"   Mapping saved to {mapping_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        
        model = MF(num_users, num_items, embed_dim=32).to(device)
        
        if os.path.exists(model_path):
            self.stdout.write(f"2. Loading existing model from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            self.stdout.write("2. Training MF Model (Quick)...")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            
            users = torch.LongTensor(df['user_idx'].values).to(device)
            items = torch.LongTensor(df['item_idx'].values).to(device)
            ratings = torch.FloatTensor(df['rating'].values).to(device)
            
            batch_size = 10000
            num_samples = len(df)
            indices = np.arange(num_samples)
            
            model.train()
            for epoch in range(5):
                np.random.shuffle(indices)
                total_loss = 0
                for i in range(0, num_samples, batch_size):
                    batch_idx = indices[i:i+batch_size]
                    u_batch = users[batch_idx]
                    i_batch = items[batch_idx]
                    r_batch = ratings[batch_idx]
                    
                    optimizer.zero_grad()
                    preds = model(u_batch, i_batch)
                    loss = criterion(preds, r_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                self.stdout.write(f"   Epoch {epoch+1}/5, Loss: {total_loss/ (num_samples/batch_size):.4f}")
            
            torch.save(model.state_dict(), model_path)
            self.stdout.write(f"   Model saved to {model_path}")

        # 3. Upload to Qdrant
        self.stdout.write("3. Uploading to Qdrant...")
        collection_name = "movies_mf"
        vector_size = 32
        
        create_collection_if_not_exists(collection_name, vector_size)
        
        # Custom Client with Timeout
        client = QdrantClient(
            host=os.getenv('QDRANT_HOST'), 
            port=int(os.getenv('QDRANT_PORT')), 
            timeout=60
        )
        
        item_embeddings = model.item_embedding.weight.detach().cpu().numpy()
        
        points = []
        batch_size_upload = 100
        total_uploaded = 0
        
        for idx, embedding in enumerate(item_embeddings):
            movie_id = idx2item.get(idx)
            if movie_id is None: continue
            
            points.append(qmodels.PointStruct(
                id=int(movie_id),
                vector=embedding.tolist(),
                payload={"movie_id": int(movie_id)}
            ))
            
            if len(points) >= batch_size_upload:
                try:
                    client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    total_uploaded += len(points)
                    points = []
                    if total_uploaded % 1000 == 0:
                        self.stdout.write(f"   Uploaded {total_uploaded} vectors...")
                except Exception as e:
                    self.stdout.write(f"   Error uploading batch: {e}")
                
        if points:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            
        self.stdout.write("\nDone! Embeddings uploaded.")
