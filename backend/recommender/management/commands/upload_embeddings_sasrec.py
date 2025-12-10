import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from django.core.management.base import BaseCommand
from django.conf import settings
from recommender.algorithms import SASRec
from recommender.vector_db import get_qdrant_client, create_collection_if_not_exists
from qdrant_client.http import models as qmodels
from qdrant_client import QdrantClient

class Command(BaseCommand):
    help = 'Train SASRec model and upload embeddings to Qdrant'

    def handle(self, *args, **kwargs):
        # 1. Load Data & Mapping
        data_dir = os.path.join(settings.BASE_DIR, 'data')
        ratings_path = os.path.join(data_dir, 'ratings.csv')
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

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.stdout.write(f"   Using device: {device}")

        # 2. SASRec Training/Loading
        self.stdout.write("\n2. Processing SASRec...")
        model_sas_path = os.path.join(settings.BASE_DIR, 'models', 'sasrec_model_retrained.pth')
        max_len = 50
        embed_dim = 64
        model_sas = SASRec(num_items + 1, max_len, embed_dim=embed_dim).to(device)
        
        if os.path.exists(model_sas_path):
            self.stdout.write(f"   Loading existing SASRec model from {model_sas_path}...")
            model_sas.load_state_dict(torch.load(model_sas_path, map_location=device))
        else:
            # Preprocess for SASRec
            df_sorted = df.sort_values(['user_idx', 'timestamp'])
            user_group = df_sorted.groupby('user_idx')['item_idx'].apply(list)
            
            optimizer = torch.optim.Adam(model_sas.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            
            model_sas.train()
            self.stdout.write("   Training SASRec (1 Epoch)...")
            
            batch_size = 128
            user_seqs = user_group.values
            indices = np.arange(len(user_seqs))
            np.random.shuffle(indices)
            
            total_loss = 0
            steps = 0
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_seqs = [user_seqs[idx] for idx in batch_idx]
                
                input_seqs = []
                target_seqs = []
                for seq in batch_seqs:
                    if len(seq) < max_len + 1:
                        pad_len = max_len + 1 - len(seq)
                        seq = [0] * pad_len + seq
                    else:
                        seq = seq[-(max_len + 1):]
                    
                    input_seqs.append(seq[:-1])
                    target_seqs.append(seq[1:])
                    
                input_tensor = torch.LongTensor(input_seqs).to(device)
                target_tensor = torch.LongTensor(target_seqs).to(device)
                
                optimizer.zero_grad()
                logits = model_sas(input_tensor)
                loss = criterion(logits.view(-1, num_items + 1), target_tensor.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                steps += 1
                if steps % 100 == 0:
                    self.stdout.write(f"   Step {steps}, Loss: {loss.item():.4f}")
            
            torch.save(model_sas.state_dict(), model_sas_path)
            self.stdout.write(f"   Model saved to {model_sas_path}")

        # 3. Upload to Qdrant
        collection_name = "movies_sasrec"
        self.stdout.write(f"\n3. Uploading to {collection_name}...")
        create_collection_if_not_exists(collection_name, embed_dim)
        
        client = QdrantClient(
            host=os.getenv('QDRANT_HOST'), 
            port=int(os.getenv('QDRANT_PORT')), 
            timeout=300
        )
        
        embeddings_sas = model_sas.item_embedding.weight.detach().cpu().numpy()
        
        points = []
        batch_size_upload = 20
        total_uploaded = 0
        
        for idx, embedding in enumerate(embeddings_sas):
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
