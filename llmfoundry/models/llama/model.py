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
        self.config = config
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
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, **kwargs):
        """Load model from HuggingFace Hub or local path."""
        print(f"Loading model from {pretrained_model_name_or_path}")
        
        # Extract custom params
        config_overrides = kwargs.pop('config_overrides', None)
        use_pretrained = kwargs.pop('pretrained', True)
        use_flash_attention_2 = kwargs.pop('use_flash_attention_2', False)
        
        # Filter out custom parameters that HF models don't accept
        for param in ['should_save_peft_only', 'shift_labels', 'peft_config', 'init_device']:
            if param in kwargs:
                kwargs.pop(param)
        
        # Load or use the provided config
        if config is None:
            from transformers import LlamaConfig
            config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Apply config overrides if provided
        if config_overrides:
            print(f"Applying config_overrides: {config_overrides}")
            for key, value in config_overrides.items():
                print(f"  Setting {key} = {value}")
                setattr(config, key, value)
        
        # Load HuggingFace model if using pretrained weights
        if use_pretrained:
            # Set flash attention if requested
            if use_flash_attention_2:
                print("Enabling Flash Attention 2")
                kwargs['attn_implementation'] = 'flash_attention_2'
            
            # Load HF model with clean kwargs
            print("Loading weights from pretrained model")
            hf_model = HFLlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
                **kwargs
            )
        else:
            print("Initializing model with random weights (pretrained=False)")
            hf_model = HFLlamaForCausalLM(config)
        
        # Initialize our custom model
        print("Creating custom LlamaForCausalLM instance")
        model = cls(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=getattr(config, 'intermediate_size', config.hidden_size * 4),
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-5),
            rope_theta=getattr(config, 'rope_theta', 500000.0),
            use_flash_attn=use_flash_attention_2,
        )
        
        # Copy weights from HF model to custom model
        if use_pretrained:
            print("Copying weights from HF model to custom model")
            model._copy_weights_from_hf_llama(hf_model)
        
        # Set config on the model
        model.config = config
        print("Model loading complete")
        
        return model

    def _copy_weights_from_hf_llama(self, hf_model):
        """Copy weights from HuggingFace model to our custom implementation"""
        # This method needs to be implemented based on your specific model structure
        # Here's a simplified version - you'll need to adapt this to your architecture
        
        # Copy embedding weights
        if hasattr(self, 'embed_tokens') and hasattr(hf_model, 'model'):
            self.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
        
        # Copy layer weights
        for i, (our_layer, hf_layer) in enumerate(zip(self.layers, hf_model.model.layers)):
            print(f"Copying weights for layer {i}/{len(self.layers)}")
            
            # Copy attention weights
            our_layer.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
            our_layer.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
            our_layer.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
            our_layer.self_attn.o_proj.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)
            
            # Copy MLP weights
            our_layer.mlp.gate_proj.weight.data.copy_(hf_layer.mlp.gate_proj.weight.data)
            our_layer.mlp.up_proj.weight.data.copy_(hf_layer.mlp.up_proj.weight.data)
            our_layer.mlp.down_proj.weight.data.copy_(hf_layer.mlp.down_proj.weight.data)
            
            # Copy layer norms
            our_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
            our_layer.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)
        
        # Copy final layer norm and lm head
        if hasattr(self, 'norm') and hasattr(hf_model.model, 'norm'):
            self.norm.weight.data.copy_(hf_model.model.norm.weight.data)
        
        if hasattr(self, 'lm_head') and hasattr(hf_model, 'lm_head'):
            self.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)

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
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # This is what we're patching in from HuggingFace's implementation
        # But having it directly in the class is cleaner
        
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # First, handle position_ids
        position_ids = kwargs.get("position_ids", None)
        
        if attention_mask is not None and position_ids is None:
            # Create position_ids based on input_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -input_ids.shape[1]:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **kwargs,
        }