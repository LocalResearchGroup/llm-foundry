import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=8192, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, position_ids: torch.LongTensor):
        # position_ids: [batch_size, seq_len]
        inv_freq = self.inv_freq.to(device=position_ids.device)
        inv_freq_expanded = inv_freq[None, None, :]  # [1, 1, dim//2]
        position_ids_expanded = position_ids[:, :, None].float()  # [batch_size, seq_len, 1]
        freqs = torch.matmul(position_ids_expanded, inv_freq_expanded)  # [batch_size, seq_len, dim//2]
        freqs = torch.cat([freqs, freqs], dim=-1)  # [batch_size, seq_len, dim]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, dim]
        sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, dim]
        return cos, sin
