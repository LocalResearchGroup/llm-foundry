from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import LlamaMLP
from .config import LlamaConfig
from .rms_norm import LlamaRMSNorm
from .decoder import LlamaDecoderLayer

from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
import os
from huggingface_hub import hf_hub_download


# TODO: clean up the rest of the code accordingly...
class LlamaForCausalLM(nn.Module):
    """Optimized Llama model for causal language modeling."""
    
    def __init__(
        self,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        num_hidden_layers=22,
        intermediate_size=None,
        vocab_size=128256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        use_unpadded_rope=True,
        use_flash_attn=True,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size or hidden_size * 4
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_unpadded_rope = use_unpadded_rope
        self.use_flash_attn = use_flash_attn
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=None)
        
        # Decoder layers - pass parameters directly
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=self.intermediate_size,
                rope_theta=rope_theta,
                use_unpadded_rope=use_unpadded_rope,
                use_flash_attn=use_flash_attn,
                rms_norm_eps=rms_norm_eps,
            ) for _ in range(num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        
        # LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
    ):
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        if return_dict:
            return {"logits": logits, "loss": loss}
        return (loss, logits)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load from Hugging Face checkpoint."""
        # Get HF Config
        hf_config = HFLlamaConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Convert to our config format
        config = LlamaConfig(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            use_cache=hf_config.use_cache,
            rope_theta=kwargs.get("rope_theta", 10000.0),
            rope_scaling=kwargs.get("rope_scaling", None),
            use_unpadded_rope=kwargs.get("use_unpadded_rope", True),
        )
        
        # Create our model
        model = cls(config)
        
        # Load HF model
        hf_model = HFLlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            torch_dtype=torch.bfloat16 if kwargs.get("bf16", True) else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Map weights from HF model to our model
        # First ensure the architecture match
        assert len(model.layers) == len(hf_model.model.layers), "Layer count mismatch"
        
        # Map embedding weights
        model.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
        model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)
        
        # Map final norm
        model.norm.weight.data.copy_(hf_model.model.norm.weight.data)
        
        # Map each layer's weights
        for i, (our_layer, hf_layer) in enumerate(zip(model.layers, hf_model.model.layers)):
            # Attention weights
            our_layer.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
            our_layer.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
            our_layer.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
            our_layer.self_attn.o_proj.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)
            
            # MLP weights
            our_layer.mlp.gate_proj.weight.data.copy_(hf_layer.mlp.gate_proj.weight.data)
            our_layer.mlp.up_proj.weight.data.copy_(hf_layer.mlp.up_proj.weight.data)
            our_layer.mlp.down_proj.weight.data.copy_(hf_layer.mlp.down_proj.weight.data)
            
            # Layer norms
            our_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
            our_layer.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)
            
        return model