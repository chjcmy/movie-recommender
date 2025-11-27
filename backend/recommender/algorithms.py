import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
    def forward(self, users, items):
        u = self.user_embedding(users)
        i = self.item_embedding(items)
        return (u * i).sum(1)

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, mlp_layers=[64, 32, 16]):
        super().__init__()
        self.gmf_user_embedding = nn.Embedding(num_users, embed_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embed_dim)
        self.mlp_user_embedding = nn.Embedding(num_users, embed_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embed_dim)
        
        mlp_modules = []
        input_size = embed_dim * 2
        for output_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_size = output_size
        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        predict_input_size = embed_dim + mlp_layers[-1]
        self.predict_layer = nn.Linear(predict_input_size, 1)
        
    def forward(self, users, items):
        gmf_u = self.gmf_user_embedding(users)
        gmf_i = self.gmf_item_embedding(items)
        gmf_out = gmf_u * gmf_i
        
        mlp_u = self.mlp_user_embedding(users)
        mlp_i = self.mlp_item_embedding(items)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=1)
        mlp_out = self.mlp_layers(mlp_in)
        
        concat = torch.cat([gmf_out, mlp_out], dim=1)
        out = self.predict_layer(concat)
        return out.squeeze()

class WideAndDeep(nn.Module):
    def __init__(self, num_users, num_items, num_genres, embed_dim=32, mlp_layers=[64, 32, 16]):
        super().__init__()
        
        # --- Wide Part ---
        self.wide_linear = nn.Linear(num_genres, 1)
        
        # --- Deep Part ---
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        
        mlp_modules = []
        input_size = embed_dim * 2
        for output_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_size = output_size
        self.deep_mlp = nn.Sequential(*mlp_modules)
        self.deep_predict = nn.Linear(input_size, 1)
        
    def forward(self, users, items, genres):
        # Wide
        wide_out = self.wide_linear(genres)
        
        # Deep
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)
        deep_in = torch.cat([u_emb, i_emb], dim=1)
        deep_features = self.deep_mlp(deep_in)
        deep_out = self.deep_predict(deep_features)
        
        # Combine
        return (wide_out + deep_out).squeeze()

class SASRec(nn.Module):
    def __init__(self, num_items, max_len, embed_dim=64, num_heads=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        
        # Embeddings
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_seq):
        # input_seq: (Batch, Max_Len)
        batch_size = input_seq.size(0)
        seq_len = input_seq.size(1)
        
        src_key_padding_mask = (input_seq == 0)
        src_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_seq.device)
        
        items = self.item_embedding(input_seq)
        positions = self.position_embedding(torch.arange(seq_len, device=input_seq.device))
        x = items + positions
        x = self.dropout(x)
        
        out = self.transformer_encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        out = self.layer_norm(out)
        
        logits = torch.matmul(out, self.item_embedding.weight.transpose(0, 1))
        
        return logits
