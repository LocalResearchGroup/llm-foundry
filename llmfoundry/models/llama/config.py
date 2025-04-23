from dataclasses import dataclass
from typing import Optional, Union, Dict, Any

@dataclass
class LlamaConfig:
    hidden_size: int = 576
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    num_hidden_layers: int = 30
    intermediate_size: int = 1536
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    vocab_size: int = 49152
    max_position_embeddings: int = 8192
    rope_theta: float = 100000.0  # Changed to float
    tie_word_embeddings: bool = False
    
    # Add these parameters for our optimizations
    use_cache: bool = True
    use_unpadded_rope: bool = True  # Control whether to use our optimized unpadded RoPE
    rope_scaling: Optional[Dict[str, Any]] = None  # For handling extended context lengths
    
    # Add parameter to control Flash Attention usage
    use_flash_attn: bool = True


# from dataclasses import dataclass

# @dataclass
# class LlamaConfig:
#     hidden_size: int = 576
#     num_attention_heads: int = 9
#     num_key_value_heads: int = 3
#     num_hidden_layers: int = 30
#     intermediate_size: int = 1536
#     hidden_act: str = "silu"
#     rms_norm_eps: float = 1e-5
#     vocab_size: int = 49152
#     max_position_embeddings: int = 8192
#     rope_theta: int = 100000
#     tie_word_embeddings: bool = False


# meta-llama/Llama-3.2-3B config:
# @dataclass
# class LlamaConfig:
#     hidden_size: int = 3072
#     num_attention_heads: int = 24
#     num_key_value_heads: int = 8
#     num_hidden_layers: int = 28
#     intermediate_size: int = 8192
#     hidden_act: str = "silu"
#     rms_norm_eps: float = 1e-5
#     vocab_size: int = 128256
#     max_position_embeddings: int = 131072
#     rope_theta: float = 500000.0
#     tie_word_embeddings: bool = True