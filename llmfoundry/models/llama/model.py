"""Custom Llama model implementation."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple, Mapping

import torch
import torch.nn as nn
from composer.models import HuggingFaceModel
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

# Add paths to Python path - use relative paths instead of hardcoded ones
current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_dir))

class CustomLlamaModel(HuggingFaceModel):
    """Custom Llama model that extends HuggingFaceModel with optimized implementation."""
    
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer: Optional[Any] = None,
        use_flash_attention_2: bool = True,
        peft_config: Optional[Dict[str, Any]] = None,
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
        **kwargs: Any
    ) -> None:
        """Initialize the custom Llama model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model
            tokenizer: Tokenizer to use
            use_flash_attention_2: Whether to use Flash Attention 2
            peft_config: Optional PEFT configuration
            hidden_size: Size of the hidden dimension
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key/value heads for grouped-query attention
            num_hidden_layers: Number of transformer layers
            intermediate_size: Size of the intermediate dimension in the MLP
            vocab_size: Size of the vocabulary
            max_position_embeddings: Maximum sequence length
            rms_norm_eps: Epsilon for RMS normalization
            rope_theta: Base for RoPE embeddings
            use_unpadded_rope: Whether to use unpadded RoPE
            use_flash_attn: Whether to use Flash Attention
            **kwargs: Additional arguments to pass to model
        """
        # Remove any parameters that might cause issues
        if 'import_path' in kwargs:
            del kwargs['import_path']
        print("✅ CUSTOM LLAMA MODEL INITIALIZED")
        # Store model configuration
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
        
        # Load the model using our custom implementation
        model = self._create_model(
            pretrained_model_name_or_path, 
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=use_flash_attention_2,
            **kwargs
        )
        
        print(f"Model type: {type(model).__name__}")
        
        # Apply PEFT if specified
        if peft_config:
            from peft import get_peft_model, LoraConfig
            peft_type = peft_config.get('peft_type', 'LORA')
            
            if peft_type == 'LORA':
                lora_config = LoraConfig(
                    r=peft_config.get('r', 8),
                    lora_alpha=peft_config.get('lora_alpha', 16),
                    lora_dropout=peft_config.get('lora_dropout', 0.05),
                    target_modules=peft_config.get(
                        'target_modules', 
                        ["q_proj", "k_proj", "v_proj", "o_proj"]
                    ),
                    bias=peft_config.get('bias', 'none'),
                    task_type=peft_config.get('task_type', 'CAUSAL_LM')
                )
                model = get_peft_model(model, lora_config)
        
        # Initialize parent class
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True
        )

    
    def _create_model(self, pretrained_model_name_or_path, **kwargs):
        """Create the model from pretrained weights or initialize from scratch."""
        # Extract custom params
        config_overrides = kwargs.pop('config_overrides', None)
        use_pretrained = kwargs.pop('pretrained', True)
        use_flash_attention_2 = kwargs.pop('use_flash_attention_2', False)
        
        # Filter out custom parameters that HF models don't accept
        for param in ['should_save_peft_only', 'shift_labels', 'peft_config', 'init_device']:
            if param in kwargs:
                kwargs.pop(param)
        
        # Load or use the provided config
        if 'config' not in kwargs:
            from transformers import LlamaConfig
            config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
        else:
            config = kwargs['config']
        
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
            
            #######
            def inspect_config(config_obj):
                """Print the config object structure and valid fields."""
                print("##############START#####################")

                print(f"Config class: {config_obj.__class__.__name__}")
                print("Config attributes:")
                
                # Get all attributes that aren't callable or private
                attrs = {attr: getattr(config_obj, attr) 
                        for attr in dir(config_obj) 
                        if not callable(getattr(config_obj, attr)) and not attr.startswith('_')}
                
                # Print in a readable format
                import json
                print(json.dumps(attrs, indent=2, default=str))
                
                # Show the config's to_dict method output if available
                if hasattr(config_obj, "to_dict") and callable(config_obj.to_dict):
                    print("\nConfig.to_dict():")
                    print(json.dumps(config_obj.to_dict(), indent=2, default=str))
                print("##############END#####################")
                return config_obj
            inspect_config(config)
            #######

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
        model = self._initialize_model_from_config(config)
        def track_computation(module, input, output):
            print(f"Module {module.__class__.__name__} called")
            print(f"  Input shapes: {[x.shape if isinstance(x, torch.Tensor) else type(x) for x in input]}")
            print(f"  Output shapes: {output.shape if isinstance(output, torch.Tensor) else [x.shape if isinstance(x, torch.Tensor) else type(x) for x in output]}")
        model.lm_head.register_forward_hook(track_computation)

        # Copy weights from HF model to custom model
        if use_pretrained:
            print("Copying weights from HF model to custom model")
            self._copy_weights_from_hf_llama(model, hf_model)
        
        # Set config on the model
        model.config = config
        print("Model loading complete")
        
        return model
    
    def _initialize_model_from_config(self, config):
        """Initialize model from config."""
        # Lazy import to avoid circular dependency
        from llmfoundry.models.llama.config import LlamaConfig
        from llmfoundry.models.llama.rms_norm import LlamaRMSNorm
        from llmfoundry.models.llama.decoder import LlamaDecoderLayer
        
        # Create a model instance
        model = nn.Module()
        
        # Create a proper LlamaConfig instance
        llama_config = LlamaConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, 'num_key_value_heads', config.num_attention_heads),
            num_hidden_layers=config.num_hidden_layers,
            intermediate_size=getattr(config, 'intermediate_size', config.hidden_size * 4),
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-5),
            rope_theta=getattr(config, 'rope_theta', 500000.0),
            use_unpadded_rope=getattr(config, 'use_unpadded_rope', True),
            use_flash_attn=getattr(config, 'use_flash_attn', True),
        )
        
        # Set model attributes from config
        model.config = llama_config
        model.hidden_size = llama_config.hidden_size
        model.num_attention_heads = llama_config.num_attention_heads
        model.num_key_value_heads = llama_config.num_key_value_heads
        model.num_hidden_layers = llama_config.num_hidden_layers
        model.intermediate_size = llama_config.intermediate_size
        model.vocab_size = llama_config.vocab_size
        model.max_position_embeddings = llama_config.max_position_embeddings
        model.rms_norm_eps = llama_config.rms_norm_eps
        model.rope_theta = llama_config.rope_theta
        model.use_unpadded_rope = llama_config.use_unpadded_rope
        model.use_flash_attn = llama_config.use_flash_attn
        
        # Embedding layer
        model.embed_tokens = nn.Embedding(model.vocab_size, model.hidden_size, padding_idx=None)
        
        # Decoder layers
        model.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(model.num_hidden_layers)
        ])
        
        # Final normalization
        model.norm = LlamaRMSNorm(model.hidden_size, eps=model.rms_norm_eps)
        
        # LM head
        model.lm_head = nn.Linear(model.hidden_size, model.vocab_size, bias=False)
        # Add flag to control whether to use fused loss
        #model._fused_loss = True
        use_fused_loss = getattr(config, 'use_fused_loss', True)  # Default to True
        model._fused_loss = use_fused_loss
        model.fused_loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)  # Add the actual loss function

        # Add forward method to the model
        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            #compute_logits = False,
            **kwargs
        ):
            # Get hidden states from embeddings
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            
            # Get position IDs if not provided
            if position_ids is None:
                position_ids = torch.arange(input_ids.size(1), device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
            # Initialize past key values if not provided
            if past_key_values is None:
                past_key_values = tuple([None] * len(self.layers))
            
            # Initialize hidden states
            hidden_states = inputs_embeds
            
            # Initialize present key values for caching
            present_key_values = () if use_cache else None
            
            # Process each layer
            for i, layer in enumerate(self.layers):
                # Get past key values for this layer
                past_key_value = past_key_values[i] if past_key_values is not None else None
                
                # Forward pass through the layer
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
                # Update hidden states
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
                
                # Store present key values if using cache
                if use_cache:
                    present_key_values += (layer_outputs[1],)
            
            # Apply final layer norm
            hidden_states = self.norm(hidden_states)
            
            # Get logits from the language model head
            #logits = self.lm_head(hidden_states)
            #
            logits = None

            # Calculate loss if labels are provided
            loss = None
            if labels is not None:
                # Get final hidden states for loss calculation
                final_hidden = hidden_states[..., :-1, :].contiguous().view(-1, self.hidden_size)
                shift_labels = labels[..., 1:].contiguous().view(-1)
                
                if hasattr(self, '_fused_loss') and self._fused_loss:
                    print("USING FUSED LOSS")
                    torch.cuda.synchronize()
                    before_mem = torch.cuda.memory_allocated()
                    loss = self.fused_loss_fn(
                        self.lm_head.weight,
                        final_hidden,
                        shift_labels
                    )
                    torch.cuda.synchronize()
                    after_mem = torch.cuda.memory_allocated()
                    print(f"Memory change during fused loss: {(after_mem - before_mem) / 1024**2:.2f} MB")
                else:
                    print("USING STANDARD LOSS")
                    torch.cuda.synchronize()
                    before_mem = torch.cuda.memory_allocated()
                    # Calculate partial logits for loss only
                    partial_logits = self.lm_head(final_hidden)
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(partial_logits, shift_labels)
                    torch.cuda.synchronize()
                    after_mem = torch.cuda.memory_allocated()
                    print(f"Memory change during standard loss: {(after_mem - before_mem) / 1024**2:.2f} MB")

            # Calculate full sequence logits ONLY if needed for generation/output
            print("=== LOGITS CALCULATION CHECK ===")
            #if return_dict or output_attentions or output_hidden_states:
            # Calculate logits if they're needed for output or generation
            if (return_dict or                    # Structured output needs logits
                not self.training or                   # Inference/generation usually needs logits
                labels is None or                 # No loss calculation means we need logits
                output_attentions or output_hidden_states):  # Special outputs need logits
                print("Full logits calculation needed for return dict/output features")
                torch.cuda.synchronize()
                before_mem = torch.cuda.memory_allocated()
                logits = self.lm_head(hidden_states)
                torch.cuda.synchronize()
                after_mem = torch.cuda.memory_allocated()
                print(f"Memory allocated for full logits: {(after_mem - before_mem) / 1024**2:.2f} MB")
            else:
                print("Skipping full logits calculation - not needed")
                # Calculate loss if labels are provided
            # loss = None
            # if labels is not None:
            #     # Track computation path
            #     print("=== LOSS CALCULATION PATH ===")
                
            #     # Get final hidden states
            #     final_hidden = hidden_states[..., :-1, :].contiguous()
            #     final_hidden_shape = final_hidden.shape
            #     print(f"Final hidden shape before view: {final_hidden_shape}")
            #     final_hidden = final_hidden.view(-1, self.hidden_size)
                
            #     shift_labels = labels[..., 1:].contiguous().view(-1)
                
            #     # Check which loss calculation path is taken
            #     if hasattr(self, '_fused_loss') and self._fused_loss:
            #         print("USING FUSED LOSS")
            #         torch.cuda.synchronize()
            #         before_mem = torch.cuda.memory_allocated()
            #         loss = self.fused_loss_fn(
            #             self.lm_head.weight,
            #             final_hidden,
            #             shift_labels,
            #             #ignore_index=-100
            #         )
            #         torch.cuda.synchronize()
            #         after_mem = torch.cuda.memory_allocated()
            #         print(f"Memory change during fused loss: {(after_mem - before_mem) / 1024**2:.2f} MB")
            #     else:
            #         print("USING STANDARD LOSS")
            #         torch.cuda.synchronize()
            #         before_mem = torch.cuda.memory_allocated()
            #         # Standard loss computation
            #         logits = self.lm_head(final_hidden)
            #         loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            #         loss = loss_fct(logits, shift_labels)
            #         torch.cuda.synchronize()
            #         after_mem = torch.cuda.memory_allocated()
            #         print(f"Memory change during standard loss: {(after_mem - before_mem) / 1024**2:.2f} MB")
            
            # # Calculate logits only if needed (check if we're actually doing this unnecessarily)
            # print("=== LOGITS CALCULATION PATH ===")
            # torch.cuda.synchronize()
            # before_mem = torch.cuda.memory_allocated()
            # if logits is None:
            #     logits = self.lm_head(hidden_states)
            # torch.cuda.synchronize()
            # after_mem = torch.cuda.memory_allocated()
            # print(f"Memory allocated for logits calculation: {(after_mem - before_mem) / 1024**2:.2f} MB")
            
            # Calculate loss if labels are provided
            # loss = None
            # if labels is not None:
            #     # Get the final hidden states (before lm_head)
            #     final_hidden = hidden_states[..., :-1, :].contiguous().view(-1, self.hidden_size)
            #     shift_labels = labels[..., 1:].contiguous().view(-1)
            #     shift_labels = shift_labels.to(final_hidden.device)
                
            #     # Use fused loss function
            #     if hasattr(self, '_fused_loss') and self._fused_loss:
            #         # For verification, print a message
            #         print("Using LigerFusedLinearCrossEntropyLoss")
            #         loss = self.fused_loss_fn(
            #             self.lm_head.weight,  # Linear weights
            #             final_hidden,         # Features before linear projection
            #             shift_labels,         # Target labels
            #             #ignore_index=-100     # Ignore padding
            #         )
            #     else:
            #         # Fallback to standard cross entropy
            #         logits = self.lm_head(final_hidden)
            #         loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            #         loss = loss_fct(logits, shift_labels)


            # Return outputs
            if return_dict:
                return {
                    "loss": loss,
                    "logits": logits,
                    "hidden_states": hidden_states,
                    "past_key_values": present_key_values,
                }
            else:
                return (loss, logits) if loss is not None else (logits,)

        # Bind the forward method to the model
        model.forward = forward.__get__(model)

        # Add prepare_inputs_for_generation method to the model
        def prepare_inputs_for_generation(
            self, 
            input_ids, 
            past_key_values=None, 
            attention_mask=None, 
            **kwargs
        ):
            # only last token for input_ids if past is not None
            if past_key_values is not None:
                input_ids = input_ids[:, -1].unsqueeze(-1)
                
                # the cache may be updated in the forward pass
                # we need to update the attention mask accordingly
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -1].unsqueeze(-1)
            
            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": kwargs.get("use_cache", True),
            }

        # Bind the method to the model
        model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model)
        
        return model
    #
    def _copy_weights_from_hf_llama(self, model, hf_model):
        """Copy weights from HuggingFace model to our custom implementation"""
        # Keep track of uncopied weights
        our_state_dict = {k: False for k in model.state_dict().keys()}
        copied_count = 0
        total_count = len(our_state_dict)
        
        # Copy embedding weights
        if hasattr(model, 'embed_tokens') and hasattr(hf_model, 'model'):
            model.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
            our_state_dict['embed_tokens.weight'] = True
            copied_count += 1
        
        # Copy layer weights
        for i, (our_layer, hf_layer) in enumerate(zip(model.layers, hf_model.model.layers)):
            print(f"Copying weights for layer {i}/{len(model.layers)}")
            
            # Copy attention weights
            layer_prefix = f"layers.{i}."
            components = [
                ('self_attn.q_proj.weight', 'self_attn.q_proj.weight'),
                ('self_attn.k_proj.weight', 'self_attn.k_proj.weight'),
                ('self_attn.v_proj.weight', 'self_attn.v_proj.weight'),
                ('self_attn.o_proj.weight', 'self_attn.o_proj.weight'),
                ('mlp.gate_proj.weight', 'mlp.gate_proj.weight'),
                ('mlp.up_proj.weight', 'mlp.up_proj.weight'),
                ('mlp.down_proj.weight', 'mlp.down_proj.weight'),
                ('input_layernorm.weight', 'input_layernorm.weight'),
                ('post_attention_layernorm.weight', 'post_attention_layernorm.weight')
            ]
            
            for our_name, hf_name in components:
                full_name = layer_prefix + our_name
                if full_name in our_state_dict:
                    # Direct copy instead of nested attribute lookup
                    our_path = our_name.split('.')
                    hf_path = hf_name.split('.')
                    
                    # Get source attribute (from HF model)
                    src = hf_layer
                    for attr in hf_path[:-1]:  # All but the last part, which is 'weight'
                        src = getattr(src, attr)
                    src_attr = getattr(src, hf_path[-1])
                    
                    # Get destination attribute (our model)
                    dst = our_layer
                    for attr in our_path[:-1]:  # All but the last part
                        dst = getattr(dst, attr)
                    dst_attr = getattr(dst, our_path[-1])
                    
                    # Copy the data
                    dst_attr.data.copy_(src_attr.data)
                    our_state_dict[full_name] = True
                    copied_count += 1
        
        # Copy final layer norm and lm head
        if hasattr(model, 'norm') and hasattr(hf_model.model, 'norm'):
            model.norm.weight.data.copy_(hf_model.model.norm.weight.data)
            our_state_dict['norm.weight'] = True
            copied_count += 1
        
        if hasattr(model, 'lm_head') and hasattr(hf_model, 'lm_head'):
            model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)
            our_state_dict['lm_head.weight'] = True
            copied_count += 1
        
        # Check for uninitialized weights
        uninitialized = [k for k, v in our_state_dict.items() if not v]
        if uninitialized:
            print(f"WARNING: {len(uninitialized)}/{total_count} weights were not initialized:")
            for name in sorted(uninitialized):
                print(f"  - {name}")
        else:
            print(f"SUCCESS: All {total_count} weights were copied successfully!")
            
        print(f"Copy rate: {copied_count}/{total_count} ({copied_count/total_count:.1%})")
    # def _copy_weights_from_hf_llama(self, model, hf_model):
    #     """Copy weights from HuggingFace model to our custom implementation"""
    #     # Copy embedding weights
    #     if hasattr(model, 'embed_tokens') and hasattr(hf_model, 'model'):
    #         model.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
        
    #     # Copy layer weights
    #     for i, (our_layer, hf_layer) in enumerate(zip(model.layers, hf_model.model.layers)):
    #         print(f"Copying weights for layer {i}/{len(model.layers)}")
            
    #         # Copy attention weights
    #         our_layer.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
    #         our_layer.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
    #         our_layer.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
    #         our_layer.self_attn.o_proj.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)
            
    #         # Copy MLP weights
    #         our_layer.mlp.gate_proj.weight.data.copy_(hf_layer.mlp.gate_proj.weight.data)
    #         our_layer.mlp.up_proj.weight.data.copy_(hf_layer.mlp.up_proj.weight.data)
    #         our_layer.mlp.down_proj.weight.data.copy_(hf_layer.mlp.down_proj.weight.data)
            
    #         # Copy layer norms
    #         our_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
    #         our_layer.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)
        
    #     # Copy final layer norm and lm head
    #     if hasattr(model, 'norm') and hasattr(hf_model.model, 'norm'):
    #         model.norm.weight.data.copy_(hf_model.model.norm.weight.data)
        
    #     if hasattr(model, 'lm_head') and hasattr(hf_model, 'lm_head'):
    #         model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> 'CustomLlamaModel':
        """Load a pretrained model."""
        return cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            **kwargs
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CustomLlamaModel':
        """Build model from config dictionary."""
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
            print(f"Model loaded successfully with {len(model.model.layers)} layers")
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
            
            # Create a dummy path for initialization
            dummy_path = "dummy_path_for_initialization"
            return cls(pretrained_model_name_or_path=dummy_path, **model_args)
    
    # def forward(self, batch: Dict[str, Any]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    #     """Custom forward method to handle the model outputs correctly.
        
    #     Args:
    #         batch: Input batch containing input_ids and labels
            
    #     Returns:
    #         Model outputs as a dictionary with 'loss' and 'logits' keys
    #     """
    #     # Filter batch to only include keys that match the model's forward arguments
    #     if isinstance(batch, Mapping):
    #         filtered_batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
    #         # Forward pass through the model
    #         outputs = self.model(**filtered_batch)
    #     else:
    #         raise ValueError(
    #             'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the model',
    #         )
        
    #     # Initialize loss and logits as None
    #     loss = None
    #     logits = None
        
    #     # Handle different output types
    #     if outputs is not None:
    #         if isinstance(outputs, tuple):
    #             # If outputs is a tuple, first element is loss, second is logits
    #             loss = outputs[0] if len(outputs) > 0 else None
    #             logits = outputs[1] if len(outputs) > 1 else None
    #         elif hasattr(outputs, 'loss') and hasattr(outputs, 'logits'):
    #             # If outputs is an object with loss and logits attributes
    #             loss = outputs.loss
    #             logits = outputs.logits
    #         else:
    #             # If outputs is just logits
    #             logits = outputs
        
    #     # Ensure we have both loss and logits
    #     if loss is None and 'labels' in batch and logits is not None:
    #         # Calculate loss if we have labels but no loss
    #         loss = torch.nn.functional.cross_entropy(
    #             logits.view(-1, logits.size(-1)),
    #             batch['labels'].view(-1),
    #             ignore_index=-100
    #         )
        
    #     return {
    #         'loss': loss,
    #         'logits': logits
    #     }
    # #
    def forward(self, batch: Dict[str, Any]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Custom forward method with diagnostic logging."""
        from torch.profiler import profile, record_function, ProfilerActivity

        if isinstance(batch, Mapping):
            filtered_batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True, record_shapes=True) as prof:
                with record_function("model_inference"):
                    outputs = self.model(**filtered_batch)
            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        else:
            raise ValueError('Unexpected batch type.')
        
        # Initialize loss and logits
        loss = None
        logits = None
        
        # Add diagnostic logging to track which branch is used
        # if outputs is not None:
        #     if isinstance(outputs, tuple):
        #         print("BRANCH: outputs is a tuple")
        #         loss = outputs[0] if len(outputs) > 0 else None
        #         logits = outputs[1] if len(outputs) > 1 else None
        #     elif hasattr(outputs, 'loss') and hasattr(outputs, 'logits'):
        #         print("BRANCH: outputs has loss and logits attributes")
        #         loss = outputs.loss
        #         logits = outputs.logits
        #     else:
        #         print("BRANCH: outputs is treated as logits")
        #         logits = outputs
        if len(outputs) > 1:
            loss, logits = outputs[0], outputs[1]
        elif len(outputs) == 1:
            loss, logits = None, outputs[0]
        else:
            loss, logits = None, None

        if loss is None and 'labels' in batch and logits is not None:
            print("BRANCH: calculating loss manually with cross_entropy")
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1),
                ignore_index=-100
            )
        
        return {'loss': loss, 'logits': logits}
    def eval_forward(self, batch: Dict[str, Any], outputs: Optional[Any] = None) -> torch.Tensor:
        """Custom eval_forward method to handle evaluation properly.
        
        Args:
            batch: Input batch containing input_ids and labels
            outputs: Optional pre-computed outputs from forward pass
            
        Returns:
            Model logits for evaluation
        """
        # If the batch mode is generate, we will generate a requested number of tokens
        if batch.get('mode', None) == 'generate':
            if self.tokenizer is None:
                raise ValueError(
                    'Generation eval cannot be used without providing a tokenizer to the model constructor.',
                )

            self.labels = batch.pop('labels')
            generation = self.generate(
                batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                synced_gpus=torch.distributed.get_world_size() > 1 if torch.distributed.is_initialized() else False,
                **batch.get('generation_kwargs', {}),
            )

            # don't remove prefix space to sentencepiece models
            if len(
                self.tokenizer(' a', add_special_tokens=False)['input_ids'],
            ) == 1:
                return self.tokenizer.batch_decode(
                    generation[:, batch['input_ids'].shape[1]:],
                    skip_special_tokens=True,
                )
            else:
                return [
                    ' ' + generation for generation in
                    self.tokenizer.batch_decode(generation[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
                ]

        # For regular evaluation or ICL task, we want to return logits
        if self.use_logits or batch.get('mode', None) == 'icl_task':
            # pop labels first to avoid computing loss
            self.labels = batch.pop('labels', None)

            # Handle encoder-decoder models
            if self.config.is_encoder_decoder and 'decoder_input_ids' not in batch and self.labels is not None:
                if hasattr(self.model, 'prepare_decoder_input_ids_from_labels'):
                    batch['decoder_input_ids'] = self.model.prepare_decoder_input_ids_from_labels(labels=self.labels)
                else:
                    raise RuntimeError(
                        'Encoder decoder models require that either decoder_input_ids is present in the batch'
                        ' or that the model has a prepare_decoder_input_ids_from_labels method.',
                    )

            # Shift labels for causal language models
            if self.shift_labels or batch.get('mode', None) == 'icl_task':
                if self.labels is not None:
                    # HF CausalLM models internally shift labels before computing loss, so we do the same here
                    self.labels[:, :-1] = self.labels[:, 1:].clone()
                    self.labels[:, -1] = -100

            # Get outputs from forward pass if not provided
            output = outputs if outputs is not None else self.forward(batch)
            
            # Extract logits from output
            if isinstance(output, dict):
                logits = output.get('logits')
            elif isinstance(output, tuple):
                # If outputs is a tuple, first element is loss, second is logits
                logits = output[1] if len(output) > 1 else output[0]
            else:
                # If outputs is just logits
                logits = output
                
            # If logits is None, return the original output
            if logits is None:
                return output
                
            # If we are in the single class case, then remove the classes dimension
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(dim=1)
                
            return logits
        else:
            # For other evaluation modes, just return the outputs
            return outputs if outputs is not None else self.forward(batch)
    
    def loss(self, outputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Dict[str, Any]) -> torch.Tensor:
        """Custom loss method to extract loss from model outputs.
        
        Args:
            outputs: Model outputs from forward method
            batch: Input batch
            
        Returns:
            Loss tensor
        """
        # If outputs is a dictionary, extract the loss
        if isinstance(outputs, dict):
            return outputs['loss']
        # If outputs is a tensor, assume it's the loss
        elif isinstance(outputs, torch.Tensor):
            return outputs
        # If outputs is a tuple, the first element is typically the loss
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            raise TypeError(f"Unexpected outputs type: {type(outputs)}")
    
    def generate(
        self, 
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.LongTensor:
        """Generate text using the model."""
        # Set evaluation mode
        self.model.eval()
        
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
                outputs = self.model(input_ids_for_step, attention_mask=attention_mask)
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
    
    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        """Return the trainable parameters of the model."""
        return [p for p in self.parameters() if p.requires_grad]

    def get_param_count(self, trainable_only: bool = False) -> int:
        """Return the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.get_trainable_params())
        else:
            return sum(p.numel() for p in self.parameters())