from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LlamaConfig:
    # Core model parameters
    hidden_size: int = 576
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    head_dim: int = 64
    num_hidden_layers: int = 30
    intermediate_size: int = 1536
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49152
    max_position_embeddings: int = 8192
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True
    
    # Attention parameters
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    
    # Token IDs
    bos_token_id: int = 0
    eos_token_id: int = 0
    pad_token_id: int = 0
    
    # Model metadata
    model_type: str = "llama"
    is_llama_config: bool = True
    initializer_range: float = 0.041666666666666664
    pretraining_tp: int = 1
    transformers_version: str = "4.40.1"
    
    # RoPE parameters
    rope_interleaved: bool = False
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Optimization parameters
    use_cache: bool = True
    use_unpadded_rope: bool = True
    use_flash_attn: bool = True
    torch_dtype: str = "bfloat16"
    
    @classmethod
    def from_smollm2_config(cls) -> 'LlamaConfig':
        """Create a LlamaConfig with smollm2 default parameters."""
        return cls(
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=0,
            eos_token_id=0,
            hidden_act="silu",
            hidden_size=576,
            initializer_range=0.041666666666666664,
            intermediate_size=1536,
            is_llama_config=True,
            max_position_embeddings=8192,
            model_type="llama",
            num_attention_heads=9,
            num_hidden_layers=30,
            num_key_value_heads=3,
            head_dim=64,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_interleaved=False,
            rope_scaling=None,
            rope_theta=100000,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            transformers_version="4.40.1",
            use_cache=True,
            vocab_size=49152
        )
