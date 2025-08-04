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

from typing import Any, Optional
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers import PreTrainedTokenizerBase
from composer.models import ComposerModel
from llmfoundry.metrics import DEFAULT_CAUSAL_LM_EVAL_METRICS, DEFAULT_CAUSAL_LM_TRAIN_METRICS
from llmfoundry.utils.builders import build_metric

from transformers import AutoModelForCausalLM

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
    vocab_size = 49152
)

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
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_output = nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=self.is_causal, 
            scale=self.scaling, enable_gqa=True).transpose(1,2)

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
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
            f"model.layers[{l}].self_attn.q_proj.weight"
        )
        model.layers[l].self_attn.k_proj.weight = assign(
            model.layers[l].self_attn.k_proj.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.layers[l].self_attn.v_proj.weight = assign(
            model.layers[l].self_attn.v_proj.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.layers[l].self_attn.o_proj.weight = assign(
            model.layers[l].self_attn.o_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.layers[l].input_layernorm.weight = assign(
            model.layers[l].input_layernorm.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Load FeedForward weights
        model.layers[l].mlp.gate_proj.weight = assign(
            model.layers[l].mlp.gate_proj.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.layers[l].mlp.up_proj.weight = assign(
            model.layers[l].mlp.up_proj.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.layers[l].mlp.down_proj.weight = assign(
            model.layers[l].mlp.down_proj.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.layers[l].post_attention_layernorm.weight = assign(
            model.layers[l].post_attention_layernorm.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    model.norm.weight = assign(model.norm.weight, params["model.norm.weight"], "model.norm.weight")
    model.lm_head.weight = assign(model.lm_head.weight, params["lm_head.weight"], "lm_head.weight")

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        if config.tie_word_embeddings:    # weight tying
            self.embed_tokens.weight = self.lm_head.weight
            
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: Any,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0))

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.lm_head(self.norm(hidden_states))

    @classmethod
    def from_pretrained(cls, model_type: str, device_map: str = "auto", torch_dtype: torch.dtype = torch.bfloat16):
        if model_type == "smollm2-135m":
            checkpoint = "HuggingFaceTB/SmolLM2-135M"
            config = SMOLLM2_CONFIG_135M
        elif model_type == "smollm2-1.7b":
            checkpoint = "HuggingFaceTB/SmolLM2-1.7B"
            raise NotImplementedError("SmolLM2-1.7B config not yet implemented")
        else:
            raise ValueError(f"Model type {model_type} not supported")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_hf = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device_map, torch_dtype=torch_dtype).to(device)
        sd_hf = model_hf.state_dict()
        model = cls(config)
        load_weights_into_smollm2(model, config, sd_hf)
        return model

class CustomLlamaModel(ComposerModel):
    """Custom Llama model wrapper for LLM Foundry compatibility."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "smollm2-135m",
        use_train_metrics: bool = True,
        additional_train_metrics: Optional[list] = None,
        additional_eval_metrics: Optional[list] = None,
        **kwargs: Any,  # Accept additional kwargs to be compatible with LLM Foundry
    ):
        super().__init__()
        if model_type == "smollm2-135m" and "model_type" in kwargs:
            model_type = kwargs["model_type"]
        
        _ = {k: v for k, v in kwargs.items() 
             if k not in ['pretrained', 'pretrained_model_name_or_path', 'name', 'model_type']}
        
        self.tokenizer = tokenizer
        self.model = LlamaModel.from_pretrained(model_type)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Build metrics
        additional_train_metrics = additional_train_metrics or []
        additional_eval_metrics = additional_eval_metrics or []
        
        train_metric_names = DEFAULT_CAUSAL_LM_TRAIN_METRICS + additional_train_metrics
        self.train_metrics = [
            build_metric(metric, {}) for metric in train_metric_names
        ] if use_train_metrics else []
        
        eval_metric_names = DEFAULT_CAUSAL_LM_EVAL_METRICS + additional_eval_metrics
        self.eval_metrics = [
            build_metric(metric, {}) for metric in eval_metric_names
        ]
    
    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        input_ids = batch['input_ids']
        outputs = self.model(input_ids=input_ids)
        return outputs
    
    # TODO: simplify loss function
    def loss(self, outputs: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        labels = batch['labels']
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    def get_metrics(self, is_train: bool = False) -> dict[str, Any]:
        metrics = self.train_metrics if is_train else self.eval_metrics
        return {metric.__class__.__name__: metric for metric in metrics}
    
    def update_metrics(self, batch: dict[str, Any], outputs: torch.Tensor, is_train: bool = False) -> None:
        metrics = self.train_metrics if is_train else self.eval_metrics
        for metric in metrics:
            metric.update(outputs, batch['labels'])
    
    def update_metric(self, batch: dict[str, Any], outputs: torch.Tensor, metric: Any) -> None:
        """Update a single metric - required by ComposerModel."""
        metric.update(outputs, batch['labels'])

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        context_size: int = 8192,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_id: Optional[int] = None
    ) -> torch.Tensor:
        self.model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = self.model(idx_cond)[:, -1, :]
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
