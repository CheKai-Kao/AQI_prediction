
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15

class ContinuousTimeEmbedding(nn.Module):
    """無使用"""
    def __init__(self, time_dim=10):
        super().__init__()
        self.time_dim = time_dim
        self.omega = nn.Parameter(torch.randn(1, time_dim))
        self.alpha = nn.Parameter(torch.randn(1, time_dim))

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(0)
        
        t = t.unsqueeze(-1)
        out = torch.sin(t * self.omega + self.alpha)
        out[..., 0] = t.squeeze(-1) * self.omega[..., 0] + self.alpha[..., 0]
        
        return out

class TTCN_Encoder(nn.Module):
    """無使用"""
    def __init__(self, d_model, seq_len=24, time_dim=10):
        super().__init__()
        self.seq_len = seq_len
        self.time_dim = time_dim
        
        self.time_enc = ContinuousTimeEmbedding(time_dim)
        
        z_dim = time_dim + 1
        
        self.meta_filter = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ReLU(),
            nn.Linear(32, d_model) 
        )
        
        self.z_projection = nn.Linear(z_dim, d_model)

    def forward(self, x, x_mask=None):
        B, V, T = x.shape
        
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        t_emb = self.time_enc(t)   # [1, 24, time_dim]
        
        t_emb = t_emb.unsqueeze(0).expand(B, V, -1, -1)   # [B, V, 24, time_dim]
        
        x_val = x.unsqueeze(-1)
        z = torch.cat([t_emb, x_val], dim=-1) # [B, V, T, time_dim + 1]
        
        filter_logits = self.meta_filter(z)  # [B, V, T, d_model]
        
        if x_mask is not None:
            mask = x_mask.unsqueeze(-1)
            filter_logits = filter_logits.masked_fill(mask == 0, -1e9)
            
        weights = F.softmax(filter_logits, dim=2) # [B, V, T, d_model]
        
        z_proj = self.z_projection(z)  # [B, V, T, d_model]
        out = torch.sum(weights * z_proj, dim=2)
        
        return out

class SparseMultiheadAttention(nn.Module):
    """
    自定義稀疏多頭注意力機制，使用 Entmax-1.5 替代 Softmax。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        B, L, _ = query.shape  # Batch, Seq_len, d_model (assume batch_first=True)
        
        q = self.q_proj(query).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = entmax15(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v) # (B, n_heads, L, d_head)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        return self.out_proj(context)


class SparseTransformerEncoderLayer(nn.Module):
    """
    整合稀疏注意力機制的編碼器層
    """
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1, activation='gelu'):
        super().__init__()
        self.self_attn = SparseMultiheadAttention(d_model, n_heads, dropout)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        #src = self.norm1(src)   # 不使用 layernorm

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        #src = self.norm2(src)
        
        return src


class iTransformer_Entmax(nn.Module):
    def __init__(self, n_vars=5, lookback_len=24, d_model=128, n_heads=2, e_layers=2, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars   # target features
        self.linear_embedding = nn.Linear(lookback_len, d_model)
        self.date_embedding = nn.Linear(5, d_model)

        self.spatial_mlp = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.encoder_layers = nn.ModuleList([
            SparseTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                activation='gelu'
            ) for _ in range(e_layers)
        ])

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 24)
        )

    def forward(self, x, x_mask, date_coord, aux_pos):
        B, _, _ = x.shape
        n_vars = self.n_vars
        device = x.device

        # Embedding
        x_enc = self.linear_embedding(x)

        # Spatial / Auxiliary Positional Encoding
        target_pos = torch.zeros((B, n_vars, 3), device=device)
        full_pos = torch.cat([target_pos, aux_pos], dim=1)
        pos_emb = self.spatial_mlp(full_pos)
        x_enc = x_enc + pos_emb

        # Date Embedding
        date_emb = self.date_embedding(date_coord)
        final_input = torch.cat([x_enc, date_emb.unsqueeze(1)], dim=1)

        # Masking Logic
        target_variate_mask = ~(x_mask.sum(dim=-1) > 0)
        date_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        final_mask = torch.cat([target_variate_mask, date_mask], dim=1)

        # Encoder Forward Pass
        enc_out = final_input
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_key_padding_mask=final_mask)

        # Projection & Output
        target_out = enc_out[:, :n_vars, :]
        output = self.projector(target_out)

        dec_out = output[:, 3, :]  # only PM2.5
        
        return dec_out


if __name__ == "__main__":
    print("🚀 開始測試模型...")
    
    BATCH = 32
    SEQ_LEN = 24
    D_MODEL = 128
    
    N_TARGET = 5
    N_AUX = 4
    TOTAL_CH = N_TARGET + N_AUX
    
    model = iTransformer_Entmax(n_vars=N_TARGET, d_model=D_MODEL)
    
    x = torch.randn(BATCH, TOTAL_CH, SEQ_LEN) # [32, 9, 24]
    x_mask = torch.ones(BATCH, TOTAL_CH, SEQ_LEN)
    aux_pos = torch.randn(BATCH, N_AUX, 3)    # [32, 4, 3]
    date_coord = torch.randn(BATCH, 5)        # [32, 5]
    
    try:
        dec_out = model(x, x_mask, date_coord, aux_pos)
        
        print(f"✅ 測試通過!")
        print(f"輸入通道數: {TOTAL_CH} (Target:{N_TARGET}, Aux:{N_AUX})")
        print(f"主輸出 (dec_out): {dec_out.shape} -> 預期 [32, 24]")
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()