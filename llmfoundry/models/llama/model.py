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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        **kwargs
    ):
        if inputs_embeds is not None and input_ids is None:
            batch_size, seq_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
            input_ids = torch.ones(batch_size, seq_length, dtype=torch.long, device=inputs_embeds.device)
        
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
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,) + (hidden_states,)
            return (loss,) + output if loss is not None else output
        
        # Return dictionary-like object
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }

    def generate(
        self, 
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        pad_token_id: int = 0,  # Remove Optional since we provide default
        eos_token_id: int = 2,  # Remove Optional since we provide default
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs) -> torch.LongTensor:
        """Generate text using the model."""
        # Set evaluation mode
        self.eval()
        
        # Create the initial sequence and attention mask
        batch_size = input_ids.shape[0]

        # Initialize generated sequences with input ids
        generated_ids = input_ids.clone()
        
        # Generate tokens one by one
        for _ in range(max_new_tokens):  # Use _ to indicate unused variable
            # Prepare inputs for current generation step
            input_ids_for_step = generated_ids
            
            # Forward pass to get logits
            with torch.no_grad():
                outputs = self(input_ids_for_step, attention_mask=attention_mask)
                next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature scaling
            if temperature > 0.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p sampling (nucleus sampling)
            if top_p < 1.0 and do_sample:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                # Scatter sorted indices to original logits
                for b in range(batch_size):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    next_token_logits[b, indices_to_remove] = -float("inf")
            
            # Sample or take argmax
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Add to generated sequence
            next_tokens = next_tokens.unsqueeze(-1)
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)
            
            # Check if all sequences have reached EOS
            if torch.all((generated_ids == eos_token_id).any(dim=1)):
                break
        
        # Ensure return type is LongTensor with explicit cast
        return generated_ids.to(dtype=torch.long)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "LlamaForCausalLM":
        """Load from Hugging Face checkpoint."""
        # Load HF model
        model_output = HFLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype=torch.bfloat16,  # Force bfloat16 
        low_cpu_mem_usage=True,
        return_dict=False)
        
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
        # If loading from pretrained, use from_pretrained directly
        pretrained_path = config.get("pretrained_model_name_or_path", None)
        if pretrained_path:
            print(f"Loading pretrained model from {pretrained_path}...")
            # Use our existing from_pretrained method with the optimizations
            model = cls.from_pretrained(
                pretrained_path,
                use_unpadded_rope=config.get("use_unpadded_rope", True),
                use_flash_attn=config.get("use_flash_attn", True),
            )
            print(f"Model loaded successfully with {len(model.layers)} layers")
            return model
            
        # Only build from scratch if no pretrained model specified
        else:
            model_args = {
                "hidden_size": config.get("d_model", 2048),
                "num_attention_heads": config.get("n_heads", 16),
                "num_key_value_heads": config.get("n_kv_heads", 4),
                "num_hidden_layers": config.get("n_layers", 22),
                "intermediate_size": config.get("d_model", 2048) * config.get("expansion_ratio", 4),
                "vocab_size": config.get("vocab_size", 128256),
                "max_position_embeddings": config.get("max_seq_len", 8192),
                "rms_norm_eps": config.get("rms_norm_eps", 1e-5),
                "rope_theta": config.get("rope_theta", 500000.0),
                "use_unpadded_rope": config.get("use_unpadded_rope", True),
                "use_flash_attn": config.get("use_flash_attn", True),
            }
            
            return cls(**model_args)

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        """Return the trainable parameters of the model."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_count(self, trainable_only: bool = False) -> int:
        """Return the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.get_trainable_params())
        else:
            return sum(p.numel() for p in self.parameters())