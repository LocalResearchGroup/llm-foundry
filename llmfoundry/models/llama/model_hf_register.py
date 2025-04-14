
from typing import Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LlamaConfig
from .rms_norm import LlamaRMSNorm
from .decoder import LlamaDecoderLayer

from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from huggingface_hub import hf_hub_download

import sys
import os
from typing import Any, Optional
from pathlib import Path
from composer.models import HuggingFaceModel
from peft import get_peft_model_state_dict, LoraConfig
from transformers import AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from llmfoundry.models.llama import LlamaForCausalLM
from llmfoundry import registry
from transformers import (
    PreTrainedTokenizerBase,
)

import copy
import inspect
import json
import logging
import random
import string
import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Union

from torchmetrics import Metric

from composer.devices import DeviceCPU
from composer.models.base import ComposerModel
from composer.utils import MissingConditionalImportError, dist, get_file, import_object, is_model_fsdp, safe_torch_load

# Add paths to Python path
sys.path.insert(0, '/llm-foundry')
sys.path.insert(0, '/llm-foundry/scripts')



try:
    from peft import PeftModel, get_peft_model
    peft_installed = True
except:
    peft_installed = False

if TYPE_CHECKING:
    import transformers
    from peft import PeftConfig, PeftModel
    from transformers import PretrainedConfig
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

log = logging.getLogger(__name__)

__all__ = ['HuggingFaceModel', 'peft_installed']


class LlamaForCausalLM(HuggingFaceModel):
    """Custom LlamaForCausalLM that inherits from HuggingFaceModel
    with compatible constructor signature."""
    
    def __init__(
        self,
        model: Union[transformers.PreTrainedModel, 'PeftModel'],
        tokenizer: Optional[Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]] = None,
        use_logits: Optional[bool] = False,
        metrics: Optional[Sequence[Metric]] = None,
        eval_metrics: Optional[Sequence[Metric]] = None,
        shift_labels: Optional[bool] = None,
        allow_embedding_resizing: bool = False,
        peft_config: Optional['PeftConfig'] = None,
        should_save_peft_only: bool = True,
        
        # Additional parameters for our custom architecture
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        num_hidden_layers: int = 22,
        intermediate_size: Optional[int] = None,
        vocab_size: int = 128256,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        use_custom_architecture: bool = False,
        custom_loss_fn: Optional[Callable] = None # allows custom loss fn
        **kwargs
    ):
        # Initialize parent with the passed model
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=use_logits,
            metrics=metrics,
            eval_metrics=eval_metrics,
            shift_labels=shift_labels,
            allow_embedding_resizing=allow_embedding_resizing,
            peft_config=peft_config,
            should_save_peft_only=should_save_peft_only,
            **kwargs
        )
        
        self.custom_loss_fn = custom_loss_fn
        # If use_custom_architecture is True, initialize our custom layers
        if use_custom_architecture:
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.num_hidden_layers = num_hidden_layers
            self.intermediate_size = intermediate_size or hidden_size * 4
            self.vocab_size = vocab_size
            
            # Create config
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
            )
            self.llama_config = config
            
            # Create our custom layers
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=None)
            
            # Decoder layers
            self.layers = nn.ModuleList([
                LlamaDecoderLayer(config) for _ in range(num_hidden_layers)
            ])
            
            # Final normalization
            self.norm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
            
            # LM head
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            
            # Replace the model's forward method with our custom implementation
            self._original_forward = self.model.forward
            #self._original_loss = self.model.loss
            self.model.forward = self._custom_forward.__get__(self.model, type(self.model))
            
            # Copy weights if the model is a LlamaForCausalLM
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                self._copy_weights_from_hf_model(model)

            self.model.forward = lambda **kwargs: self._custom_forward(**kwargs)
        else:
            self.model.forward = super().model.forward
            #self.model.loss = super().model.loss
    # def forward(self, batch):
    #     """Override parent's forward method."""
    #     if isinstance(batch, Mapping):
    #         # Filter batch to valid args (same as HuggingFaceModel)
    #         batch = {k: v for k, v in batch.items() if k in self.model_forward_args}
            
    #         # If using custom architecture, use our implementation
    #         if self.using_custom_architecture:
    #             return self._custom_forward_impl(**batch)
    #         else:
    #             # Otherwise use the original model's forward
    #             return self.model(**batch)
    #     else:
    #         raise ValueError(
    #             'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model',
    #         )
    def _custom_forward(self, batch):
        """Combined method that handles batch processing and custom model implementation"""
        # --- PART 1: Handle different batch formats (from your forward method) ---
        if isinstance(batch, dict):
            # Extract inputs for our custom implementation
            input_ids = batch.get('input_ids')
            attention_mask = batch.get('attention_mask')
            position_ids = batch.get('position_ids')
            labels = batch.get('labels')
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            inputs = batch[0]
            if isinstance(inputs, dict):
                input_ids = inputs.get('input_ids')
                attention_mask = inputs.get('attention_mask')
                position_ids = inputs.get('position_ids')
                labels = inputs.get('labels')
            elif torch.is_tensor(inputs):
                input_ids = inputs
                attention_mask = None
                position_ids = None
                labels = batch[1] if len(batch) > 1 else None
            else:
                raise TypeError(f"Unsupported input type: {type(inputs)}")
        elif torch.is_tensor(batch):
            input_ids = batch
            attention_mask = None
            position_ids = None
            labels = None
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        
        # --- PART 2: Custom model implementation ---
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
        
        # # Calculate loss if labels are provided
        # loss = None
        # loss_fct = self.custom_loss_fn if self.custom_loss_fn else torch.nn.CrossEntropyLoss()
        # if labels is not None:
        #     # Standard causal LM loss
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     #loss_fct = torch.nn.CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        # # Return in the format that works with your existing loss method
        # #from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
        # # return CausalLMOutputWithCrossAttentions(
        # #     loss=loss,
        # #     logits=logits,
        # #     hidden_states=hidden_states
        # # )
        
        # Return dictionary-like object
        # return {
        #     'loss': loss,
        #     'logits': logits,
        #     'hidden_states': hidden_states
        # }
        return {
        'logits': logits,
        'hidden_states': hidden_states,
        'labels': labels  # Include labels for the loss method
    }
    def loss(self, outputs, batch):
        """Comprehensive loss computation with robust error handling."""
        # STEP 1: Extract logits from outputs (handling various formats)
        logits = None
        if isinstance(outputs, tuple):
            # Handle tuple outputs
            if len(outputs) >= 1:
                logits = outputs[0]
        elif hasattr(outputs, 'logits'):
            # Handle object with attributes (like HuggingFace outputs)
            logits = outputs.logits
        elif isinstance(outputs, dict) or hasattr(outputs, 'get'):
            # Handle dictionary outputs or dict-like objects
            logits = outputs.get('logits')
        
        # Verify logits were extracted
        if logits is None:
            print(f"WARNING: Could not extract logits from outputs type: {type(outputs)}")
            return torch.tensor(1.0, requires_grad=True, device=self._get_device())
        
        # Ensure logits has dimensions
        if not hasattr(logits, 'ndim') or logits.ndim == 0:
            print(f"WARNING: Invalid logits shape: {getattr(logits, 'shape', 'unknown')}")
            return torch.tensor(1.0, requires_grad=True, device=logits.device if hasattr(logits, 'device') else self._get_device())
        
        # STEP 2: Extract labels
        labels = None
        # Check for labels in outputs
        if isinstance(outputs, dict) and 'labels' in outputs:
            labels = outputs['labels']
        elif hasattr(outputs, 'labels'):
            # Try attribute access if available
            labels = outputs.labels
        # Check for labels in batch
        elif isinstance(batch, dict) and 'labels' in batch:
            labels = batch['labels']
        elif isinstance(batch, (list, tuple)) and len(batch) > 1:
            _, labels = batch
        
        # Verify labels were extracted
        if labels is None or (hasattr(labels, 'numel') and labels.numel() == 0):
            print("WARNING: Could not extract valid labels")
            return torch.tensor(1.0, requires_grad=True, device=logits.device)
        
        # STEP 3: Compute loss with full error handling
        try:
            # Handle causal language modeling (shifted labels)
            if logits.size(1) != 1 and labels.size(1) != 1 and logits.size(1) == labels.size(1):
                # For sequence modeling, shift the prediction and target
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Use custom loss function if available, otherwise use CrossEntropyLoss
                loss_fn = getattr(self, 'custom_loss_fn', None) or torch.nn.CrossEntropyLoss()
                
                # Reshape for loss computation
                return loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            else:
                # For non-shifted cases
                loss_fn = getattr(self, 'custom_loss_fn', None) or torch.nn.CrossEntropyLoss()
                return loss_fn(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1)
                )
        except Exception as e:
            print(f"ERROR computing loss: {e}")
            if hasattr(logits, 'shape') and hasattr(labels, 'shape'):
                print(f"Shapes - logits: {logits.shape}, labels: {labels.shape}")
            # Return fallback loss
            return torch.tensor(1.0, requires_grad=True, device=logits.device)
    
    def _get_device(self):
        """Helper method to get a device for tensor creation"""
        if hasattr(self, 'model') and hasattr(self.model, 'device'):
            return self.model.device
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
        
    def _copy_weights_from_hf_model(self, hf_model):
        """Copy weights from HuggingFace model to our custom layers"""
        self.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
        
        for i, (our_layer, hf_layer) in enumerate(zip(self.layers, hf_model.model.layers)):
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
        self.norm.weight.data.copy_(hf_model.model.norm.weight.data)
        self.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)

def register_model():
    """Register our custom model with llm-foundry"""
    print("=== 🔧 Registering CustomLlamaModel as hf_causal_lm ===")
    # Register the model under the same name used in YAML configs
    registry.models.register("hf_causal_lm")(LlamaForCausalLM)
    print("=== 🔧 CustomLlamaModel successfully registered as hf_causal_lm - REGISTRATION COMPLETE ===")
    return True


# Register the model when this module is imported
register_model()