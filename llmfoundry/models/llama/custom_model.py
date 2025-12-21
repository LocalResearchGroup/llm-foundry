# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

from typing import Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from peft import PeftModel
from llmfoundry.models.hf.hf_base import BaseHuggingFaceModel
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from transformers.utils import is_flash_attn_2_available
from llmfoundry.data.finetuning.collator import CROSS_ENTROPY_IGNORE_INDEX

from transformers import AutoModelForCausalLM

if is_flash_attn_2_available():
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

SMOLLM2_CONFIG_135M = LlamaConfig(
    attention_bias = False,
    attention_dropout = 0.0,
    bos_token_id = 0,
    eos_token_id = 0,
    head_dim = 64,
    hidden_act = "silu",
    hidden_size = 576,
    initializer_range = 0.041666666666666664,
    intermediate_size = 1536,
    is_llama_config = True,
    max_position_embeddings = 8192,
    mlp_bias = False,
    model_type = "llama",
    num_attention_heads = 9,
    num_hidden_layers = 30,
    num_key_value_heads = 3,
    pretraining_tp = 1,
    rms_norm_eps = 1e-05,
    rope_interleaved = False,
    rope_scaling = None,
    rope_theta = 100000,
    tie_word_embeddings = True,
    torch_dtype = "bfloat16",
    transformers_version = "4.55.0.dev0",
    use_cache = True,
    vocab_size = 49152,
    _attn_implementation = "sdpa",
    _use_liger_rms_norm = False,
    _use_liger_fused_crossentropy = False,
)

# Modernbert unpadding and repadding
def _unpad_modernbert_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Remove padding from input sequences.

    Args:
        inputs: (batch, seqlen, ...) or (batch, seqlen)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        position_ids: (batch, seqlen), int, position ids
        labels: (batch, seqlen), int, labels

    Returns:
        unpadded_inputs: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths
        max_seqlen_in_batch: int
        unpadded_position_ids: (total_nnz) or None
        unpadded_labels: (total_nnz) or None
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = position_ids.flatten()[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels

def _pad_modernbert_output(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """Add padding to sequences.

    Args:
        inputs: (total_nnz, ...) or (total_nnz,), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        batch: int, batch size
        seqlen: int, max sequence length

    Returns:
        padded_inputs: (batch, seqlen, ...) or (batch, seqlen)
    """
    if inputs.dim() == 1:
        output = torch.zeros(batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    return padded_inputs

class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # (total_nnz, nheads, headdim)
        apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=False,
            inplace=True,
        )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return x

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=False,
            inplace=True,
            conjugate=True,
        )

        return do, None, None, None, None, None, None


def apply_rotary_unpadded(
    x,
    cos,
    sin,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """Arguments:
        x: (total_nnz, nheads, headdim) - input tensor for packed QKV.
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (total_nnz, dim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """  # noqa: D205
    return ApplyRotaryEmbUnpad.apply(x, cos, sin, cu_seqlens, max_seqlen)


class ModernBertUnpaddedRotaryEmbedding(RotaryEmbedding):
    """The rotary position embeddings applied directly to unpadded sequences."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seqlen: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """max_seqlen: if max_seqlen, device, and dtype are provided, we precompute the cos_sin_cache
        up to max_seqlen. If the max_seqlen, device, or dtype during training/inference differ,
        the cos_sin_cache will be recomputed during the forward pass. 
        """  # noqa: D205
        super().__init__(dim=dim, base=base, device=device, interleaved=False)
        self.max_seqlen = max_seqlen

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Apply rotary embedding *inplace* to x.
        x: (total_nnz, nheads, headdim)
        cu_seqlens: (batch + 1,) cumulative sequence lengths
        max_seqlen: int max seq length in the batch
        """  # noqa: D205
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=x.device, dtype=x.dtype)

        x = apply_rotary_unpadded(
            x,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, scale_base={self.scale_base}"

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.rope_type = "default"
        self.config = config
        head_dim = self.config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0,head_dim,2)[:(head_dim//2)].float()/head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()  # type: ignore
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float().to(x.device)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def flash_attention_forward(
    query_states: torch.Tensor, 
    key_states: torch.Tensor, 
    value_states: torch.Tensor, 
    is_causal: bool, 
    scaling: float, 
    enable_gqa: bool, 
    input_shape: tuple[int, int], 
    hidden_shape: tuple[int, int], 
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    rotary_emb: Union[LlamaRotaryEmbedding, ModernBertUnpaddedRotaryEmbedding],
    target_dtype: torch.dtype = torch.bfloat16,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    # (total_seqlen, nheads, headdim)
    query_states = query_states.view(hidden_shape)
    key_states = key_states.view(hidden_shape)
    value_states = value_states.view(hidden_shape)

    query_states = rotary_emb(query_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    key_states = rotary_emb(key_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    convert_dtype = query_states.dtype not in (torch.float16, torch.bfloat16)

    if convert_dtype:
        # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
        # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
        orig_dtype = query_states.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

        attn = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            deterministic=False,
            causal=is_causal,
        )
        attn = attn.to(orig_dtype)  # type: ignore
    else:
        attn = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0.0,
            deterministic=False,
            causal=is_causal,
        )
    total_tokens = attn.shape[0]
    hidden_size = attn.shape[1] * attn.shape[2]  # num_heads * head_dim
    return attn.view(total_tokens, hidden_size)

def sdpa_attention_forward(
    query_states: torch.Tensor, 
    key_states: torch.Tensor, 
    value_states: torch.Tensor, 
    is_causal: bool, 
    scaling: float, 
    enable_gqa: bool, 
    input_shape: tuple[int, int], 
    hidden_shape: tuple[int, int], 
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
    **kwargs: Any,
) -> torch.Tensor:
    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attn_output = nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=is_causal, 
            scale=scaling, enable_gqa=enable_gqa).transpose(1,2)

    attn_output = attn_output.reshape(*input_shape, -1)
    return attn_output

LLAMA_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
    "sdpa": sdpa_attention_forward,
}

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads*self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads*self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads*self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads*self.head_dim, config.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        rotary_emb: Optional[Union[LlamaRotaryEmbedding, ModernBertUnpaddedRotaryEmbedding]] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        attn_output = LLAMA_ATTENTION_FUNCTION[self.config._attn_implementation](
            query_states, key_states, value_states, self.is_causal, self.scaling, 
            True, input_shape, hidden_shape, position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_emb=rotary_emb,
            **kwargs,
        )

        attn_output = self.o_proj(attn_output)
        return attn_output

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        use_liger_rms_norm = getattr(config, '_use_liger_rms_norm', False)
        norm_cls = LigerRMSNorm if use_liger_rms_norm else LlamaRMSNorm
        self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        rotary_emb: Optional[Union[LlamaRotaryEmbedding, ModernBertUnpaddedRotaryEmbedding]] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            rotary_emb=rotary_emb,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

def assign(left: torch.Tensor, right: torch.Tensor, tensor_name: str = "unknown") -> torch.nn.Parameter:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_smollm2(model: 'LlamaModel', param_config: LlamaConfig, params: dict[str, torch.Tensor]) -> None:
    model.embed_tokens.weight = assign(model.embed_tokens.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    
    for l in range(param_config.num_hidden_layers):
        model.layers[l].self_attn.q_proj.weight = assign(
            model.layers[l].self_attn.q_proj.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers[{l}].self_attn.q_proj.weight",
        )
        model.layers[l].self_attn.k_proj.weight = assign(
            model.layers[l].self_attn.k_proj.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight",
        )
        model.layers[l].self_attn.v_proj.weight = assign(
            model.layers[l].self_attn.v_proj.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight",
        )
        model.layers[l].self_attn.o_proj.weight = assign(
            model.layers[l].self_attn.o_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight",
        )
        model.layers[l].input_layernorm.weight = assign(
            model.layers[l].input_layernorm.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight",
        )

        # Load FeedForward weights
        model.layers[l].mlp.gate_proj.weight = assign(
            model.layers[l].mlp.gate_proj.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight",
        )
        model.layers[l].mlp.up_proj.weight = assign(
            model.layers[l].mlp.up_proj.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight",
        )
        model.layers[l].mlp.down_proj.weight = assign(
            model.layers[l].mlp.down_proj.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight",
        )
        model.layers[l].post_attention_layernorm.weight = assign(
            model.layers[l].post_attention_layernorm.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight",
        )

    model.norm.weight = assign(model.norm.weight, params["model.norm.weight"], "model.norm.weight")
    model.lm_head.weight = assign(model.lm_head.weight, params["lm_head.weight"], "lm_head.weight")

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        use_liger_rms_norm = getattr(config, '_use_liger_rms_norm', False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        norm_cls = LigerRMSNorm if use_liger_rms_norm else LlamaRMSNorm
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._use_liger_fused_crossentropy = getattr(config, '_use_liger_fused_crossentropy', False)
        self.can_generate = True
        self.tie_weights()
            
    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        batch_size, seq_len = input_ids.shape[:2]
        repad = False
        cu_seqlens = None
        max_seqlen = None
        indices = None
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is None:
                attention_mask = (input_ids != self.config.eos_token_id).long()
            if inputs_embeds is None:
                repad = True
                with torch.no_grad():
                    input_ids, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                        inputs=input_ids, attention_mask=attention_mask,
                    )
        else:
            attention_mask = None

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        # For flash attention, we don't need position_embeddings since ModernBertUnpaddedRotaryEmbedding handles it
        if self.config._attn_implementation == "flash_attention_2":
            device = hidden_states.device if hidden_states is not None else input_ids.device
            self.rotary_emb = ModernBertUnpaddedRotaryEmbedding(
                dim=self.config.head_dim, 
                base=self.config.rope_theta,
                max_seqlen=self.config.max_position_embeddings,
                device=device,
            )
            position_embeddings = None
        else:
            device = hidden_states.device if hidden_states is not None else input_ids.device
            self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
            position_embeddings = self.rotary_emb(hidden_states, torch.arange(seq_len, device=device).unsqueeze(0))

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                rotary_emb=self.rotary_emb,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        if repad:
            hidden_states = _pad_modernbert_output(
                inputs=hidden_states, indices=indices, batch=batch_size, seqlen=seq_len,
            )
        if self._use_liger_fused_crossentropy: return hidden_states
        else: return self.lm_head(hidden_states)
 
    @classmethod
    def from_pretrained(cls, model_type: str, device_map: str = "auto", torch_dtype: torch.dtype = torch.bfloat16, use_liger_rms_norm: bool = False, use_liger_fused_crossentropy: bool = False):
        if model_type == "smollm2-135m":
            checkpoint = "HuggingFaceTB/SmolLM2-135M"
            config = SMOLLM2_CONFIG_135M
        elif model_type == "smollm2-1.7b":
            checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
            raise NotImplementedError("SmolLM2-1.7B config not yet implemented")
        else:
            raise ValueError(f"Model type {model_type} not supported")
        config._use_liger_rms_norm = use_liger_rms_norm
        config._use_liger_fused_crossentropy = use_liger_fused_crossentropy
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_hf = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device_map, torch_dtype=torch_dtype).to(device)
        sd_hf = model_hf.state_dict()
        model = cls(config)
        load_weights_into_smollm2(model, config, sd_hf)
        return model   

    # Some functions required for HuggingFace compatibility
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[tuple] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare inputs for generation. Required by PEFT and HuggingFace generation utilities."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        })
        
        return model_inputs

    def get_input_embeddings(self) -> nn.Embedding: return self.embed_tokens

    def get_output_embeddings(self) -> nn.Linear: return self.lm_head

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.embed_tokens.weight = self.lm_head.weight

    def get_decoder(self): return self

class CustomLlamaModel(BaseHuggingFaceModel):
    """Custom Llama model wrapper for LLM Foundry compatibility."""
    
    _use_liger_rms_norm: bool = False
    _use_liger_fused_crossentropy: bool = False
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "smollm2-135m",
        pretrained: bool = True,
        use_liger_rms_norm: bool = False,
        use_liger_fused_crossentropy: bool = False,
        peft_config: Optional[dict[str, Any]] = None,
        pretrained_model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M",
        **kwargs: Any,
    ):
        CustomLlamaModel._use_liger_rms_norm = use_liger_rms_norm
        CustomLlamaModel._use_liger_fused_crossentropy = use_liger_fused_crossentropy
        self._use_liger_fused_crossentropy = use_liger_fused_crossentropy
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer=tokenizer,
            pretrained=pretrained,
            peft_config=peft_config,
            shift_labels=True,
            **kwargs,
        )
        if use_liger_fused_crossentropy: self.liger_loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=CROSS_ENTROPY_IGNORE_INDEX, reduction='mean')
    
    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(input_ids=batch['input_ids'])

    def loss(self, outputs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        targets = torch.roll(batch['labels'], shifts=-1, dims=1)
        targets[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
        targets_flat = targets.flatten()
        outputs_flat = outputs.flatten(0, -2)
        
        if self._use_liger_fused_crossentropy:
            return self.liger_loss_fn(self.model.lm_head.weight, outputs_flat, targets_flat)
        else:
            return F.cross_entropy(outputs_flat, targets_flat, ignore_index=CROSS_ENTROPY_IGNORE_INDEX)

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        context_size: int = 8192,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        model = self.model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)[:, -1, :]
            if top_k > 0:
                top_logits, _ = torch.topk(logits, top_k)
                logits = torch.where(logits < top_logits[:, -1], torch.tensor(float('-inf')).to(logits.device), logits)
            if temperature > 0.0:
                idx_next = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_id is not None and idx_next == eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

    # Below are the methods that are required for PEFT compatibility
    @property
    def device(self) -> torch.device: return next(self.model.parameters()).device

    def transform_model(self, model: PreTrainedModel) -> PreTrainedModel: return model
    
    # TODO: Use config_overrides to load the model
    @classmethod
    def build_inner_model(
        cls,
        pretrained_model_name_or_path: str,
        pretrained_lora_id_or_path: Optional[str] = None,
        trust_remote_code: bool = True,
        init_device: str = 'cpu',
        use_flash_attention_2: bool = True,
        use_auth_token: bool = False,
        config_overrides: Optional[dict[str, Any]] = None,
        load_in_8bit: bool = False,
        pretrained: bool = True,
        prepare_for_fsdp: bool = False,
        **kwargs: Any,
    ) -> Union[PreTrainedModel, 'PeftModel']:
        """Build your custom model instead of using AutoModelForCausalLM."""
        use_liger_rms_norm = cls._use_liger_rms_norm
        use_liger_fused_crossentropy = cls._use_liger_fused_crossentropy
        if pretrained:
            model = LlamaModel.from_pretrained("smollm2-135m", use_liger_rms_norm=use_liger_rms_norm, use_liger_fused_crossentropy=use_liger_fused_crossentropy)
            model.config._use_liger_fused_crossentropy = use_liger_fused_crossentropy
        else:
            from copy import deepcopy
            config = deepcopy(SMOLLM2_CONFIG_135M)
            config._use_liger_rms_norm = use_liger_rms_norm
            config._use_liger_fused_crossentropy = use_liger_fused_crossentropy
            if config_overrides:
                for key, value in config_overrides.items():
                    setattr(config, key, value)
            model = LlamaModel(config)
        
        if pretrained_lora_id_or_path is not None:
            from composer.models.huggingface import peft_installed
            if not peft_installed:
                raise ValueError(
                    'PEFT is not installed, but lora_id_or_path was passed. Please install LLM Foundry with the peft extra to use lora_id_or_path.',
                )
            from peft import PeftModelForCausalLM
            model = PeftModelForCausalLM.from_pretrained(
                model,
                pretrained_lora_id_or_path,
                is_trainable=True,
            )
        
        if prepare_for_fsdp:
            cls.prepare_inner_model(model, init_device)
        return model

    def eval_forward(self, batch: dict[str, Any], outputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if outputs is not None:
            return outputs
        else:
            return self.forward(batch)

    def update_metric(self, batch: dict[str, Any], outputs: torch.Tensor, metric: Any) -> None:
        metric.update(outputs, batch['labels'])