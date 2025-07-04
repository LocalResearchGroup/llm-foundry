from flash_attn import flash_attn_func
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .liger_rope import LigerRopeFunction
from .config import LlamaConfig

class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_attention_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.register_buffer(
            "cos_cached",
            self._compute_rope_embeddings(
                self.max_position_embeddings,
                self.head_dim,
                self.rope_theta,
                dtype=torch.float32,
                device=self.q_proj.weight.device,
            )[0],
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            self._compute_rope_embeddings(
                self.max_position_embeddings,
                self.head_dim,
                self.rope_theta,
                dtype=torch.float32,
                device=self.q_proj.weight.device,
            )[1],
            persistent=False,
        )

    def _compute_rope_embeddings(self, max_position_embeddings, head_dim, base=10000, dtype=None, device=None):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(max_position_embeddings, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos.unsqueeze(0), sin.unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # In B S (H D)
        bsz, seq_len, _ = hidden_states.size()
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
            position_ids = repeat(position_ids, 'l -> b l', b=bsz)
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = rearrange(query_states, "b s (h d) -> b s h d", h=self.num_heads, d=self.head_dim)
        key_states = rearrange(key_states, "b s (h d) -> b s h d", h=self.num_key_value_heads, d=self.head_dim)
        value_states = rearrange(value_states, "b s (h d) -> b s h d", h=self.num_key_value_heads, d=self.head_dim)

        # Slice off position specific rope freqs from the cached freqs
        cos = self.cos_cached[:, position_ids]  # [1, bsz, seq_len, dim]
        sin = self.sin_cached[:, position_ids]  # [1, bsz, seq_len, dim]
        
        query_states, key_states = LigerRopeFunction.apply(
            query_states,
            key_states,
            cos.squeeze(0),
            sin.squeeze(0),
            position_ids
        )

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=0.0,
            causal=attention_mask is None
        )
        
        attn_output = rearrange(attn_output, "b s h d -> b s (h d)")
        return self.o_proj(attn_output)