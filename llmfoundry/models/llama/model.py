from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LlamaConfig
from .rms_norm import LlamaRMSNorm
from .decoder import LlamaDecoderLayer

from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from huggingface_hub import hf_hub_download


# TODO: clean up the rest of the code accordingly...
class LlamaForCausalLM(nn.Module):
    """Optimized Llama model for causal language modeling."""
    
    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        num_hidden_layers: int = 22,
        intermediate_size: Optional[int] = None,
        vocab_size: int = 128256,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        use_unpadded_rope: bool = True,
        use_flash_attn: bool = True,
        **kwargs
    ) -> None:
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
        
        # Create a config for the decoder layers
        config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=self.intermediate_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            use_unpadded_rope=use_unpadded_rope,
            use_flash_attn=use_flash_attn,
        )
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=None)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(num_hidden_layers)
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
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "LlamaForCausalLM":
        """Load from Hugging Face checkpoint."""
        # Load HF model
        model_output = HFLlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            torch_dtype=torch.bfloat16 if kwargs.get("bf16", True) else torch.float32,
            low_cpu_mem_usage=True,
            return_dict=False
        )
        
        # Handle tuple return - some HF methods return (model, loading_info)
        if isinstance(model_output, tuple):
            hf_model = model_output[0]
        else:
            hf_model = model_output
        
        # Extract config from HF model
        hf_config = hf_model.config
        
        # Create our model with matching architecture
        model = cls(
            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
            num_hidden_layers=hf_config.num_hidden_layers,
            intermediate_size=hf_config.intermediate_size,
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
            use_unpadded_rope=kwargs.get("use_unpadded_rope", True),
            use_flash_attn=kwargs.get("use_flash_attn", True),
        )
        
        # Map weights from HF model to our model
        # First ensure the architecture matches
        assert len(model.layers) == len(hf_model.model.layers), "Layer count mismatch"
        
        # Map embedding weights
        model.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
        model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)
        
        # Map final norm
        model.norm.weight.data.copy_(hf_model.model.norm.weight.data)
        
        # Map each layer's weights
        for layer_idx, (our_layer, hf_layer) in enumerate(zip(model.layers, hf_model.model.layers)):
             print(f"Copying weights for layer {layer_idx}/{len(model.layers)}")
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

    @classmethod
    def from_config(cls, config: dict) -> "LlamaForCausalLM":
        """Build model from llm-foundry config dictionary."""
        # Map llm-foundry config keys to our init parameters
        model_args = {
            "hidden_size": config.get("d_model", 2048),
            "num_attention_heads": config.get("n_heads", 16),
            "num_key_value_heads": config.get("n_kv_heads", 4),
            "num_hidden_layers": config.get("n_layers", 22),
            "intermediate_size": config.get("intermediate_size", None),  # Will default to 4*hidden_size
            "vocab_size": config.get("vocab_size", 128256),
            "max_position_embeddings": config.get("max_seq_len", 8192),
            "rms_norm_eps": config.get("rms_norm_eps", 1e-5),
            "rope_theta": config.get("rope_theta", 500000.0),
            "use_unpadded_rope": config.get("use_unpadded_rope", True),
            "use_flash_attn": config.get("use_flash_attn", True),
        }
        
        # Create model instance
        model = cls(**model_args)
        
        # Load pretrained weights if specified
        pretrained_path = config.get("pretrained_model_name_or_path", None)
        if pretrained_path:
            pretrained_model = cls.from_pretrained(
                pretrained_path,
                use_unpadded_rope=model_args["use_unpadded_rope"],
                use_flash_attn=model_args["use_flash_attn"],
            )
            # Copy weights from pretrained model to our model
            model.load_state_dict(pretrained_model.state_dict())
        
        return model

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        """Return the trainable parameters of the model."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_count(self, trainable_only: bool = False) -> int:
        """Return the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.get_trainable_params())
        else:
            return sum(p.numel() for p in self.parameters())