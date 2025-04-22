import sys
import os
import torch
from typing import Any, Optional
from pathlib import Path
from composer.models import HuggingFaceModel, ComposerModel
from peft import get_peft_model_state_dict, LoraConfig
from transformers import AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from llmfoundry.models.llama import LlamaForCausalLM
from llmfoundry import registry
from transformers import (
    PreTrainedTokenizerBase,
)
# Add paths to Python path
sys.path.insert(0, '/llm-foundry')
sys.path.insert(0, '/llm-foundry/scripts')

# First define the CustomLlamaModel class
class CustomLlamaModel(HuggingFaceModel):
    """Custom Llama model with PEFT adapter support"""
    #
    def __init__(
    self,
    tokenizer: PreTrainedTokenizerBase,
    pretrained_model_name_or_path: str,
    pretrained: bool = True,
    pretrained_lora_id_or_path: Optional[str] = None,
    trust_remote_code: bool = True,
    use_auth_token: bool = False,
    use_flash_attention_2: bool = False,
    load_in_8bit: bool = False, 
    init_device: str = 'cpu',
    config_overrides: Optional[dict[str, Any]] = None,
    peft_config: Optional[dict[str, Any]] = None,
    use_train_metrics: bool = True,
    allow_embedding_resizing: bool = False,
    additional_train_metrics: Optional[list] = None,
    additional_eval_metrics: Optional[list] = None,
    should_save_peft_only: bool = True,
    **kwargs
):
        print("=== Initializing CustomLlamaModel ===")
        
        # Initialize config
        config = None
        if pretrained:
            print(f"Loading pretrained config from {pretrained_model_name_or_path}")
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code
            )
        else:
            print("Initializing new config")
            config = AutoConfig.for_model("llama")
            
        # Apply config overrides if provided
        if config_overrides:
            print(f"Applying config overrides: {config_overrides}")
            for key, value in config_overrides.items():
                setattr(config, key, value)
        
        # Initialize the model
        model_kwargs = {
            "config": config,
            "torch_dtype": torch.bfloat16
        }
        
        if use_flash_attention_2:
            print("Using Flash Attention 2")
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        if use_auth_token:
            model_kwargs["use_auth_token"] = use_auth_token
        
        # Create the model using our custom LlamaForCausalLM implementation
        print(f"Creating model from {pretrained_model_name_or_path}")
        if pretrained:
            hf_model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                **model_kwargs
            )
        else:
            hf_model = LlamaForCausalLM(config)
        
        print(f"Created model of type: {type(hf_model).__name__}")
        
        # Process PEFT config
        if isinstance(peft_config, dict):
            print("Converting PEFT config dict to LoraConfig")
            peft_type = peft_config.get('peft_type', 'LORA').upper()
            if peft_type == 'LORA':
                lora_config = {
                    'r': peft_config.get('r', 8),
                    'lora_alpha': peft_config.get('lora_alpha', 16),
                    'lora_dropout': peft_config.get('lora_dropout', 0.05),
                    'target_modules': peft_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
                    'bias': 'none',
                    'task_type': peft_config.get('task_type', 'CAUSAL_LM'),
                    'base_model_name_or_path': pretrained_model_name_or_path
                }
                print(f"Creating LoraConfig with: {lora_config}")
                peft_config = LoraConfig(**lora_config)
        
        # Call parent HuggingFaceModel constructor
        print("Initializing HuggingFaceModel")
        super().__init__(
            model=hf_model,
            tokenizer=tokenizer,
            use_logits=False,
            metrics=additional_train_metrics,
            eval_metrics=additional_eval_metrics,
            shift_labels=True,  # For causal LM
            allow_embedding_resizing=allow_embedding_resizing,
            peft_config=peft_config,
            should_save_peft_only=should_save_peft_only
        )
        
        # Store additional attributes we need
        self.peft_config = peft_config
        self.using_peft = peft_config is not None
        self.should_save_peft_only = should_save_peft_only
        
        # Initialize counter for monitoring
        self.forward_count = 0
        
        print("CustomLlamaModel initialization complete")

    #
    def forward(self, batch):
        """Forward method that properly handles Composer batch format"""
        # Determine if input is a dictionary batch (from Composer) or direct tensors
        if isinstance(batch, dict):
            return self.model(**batch)
        
        elif isinstance(batch, (list, tuple)) and len(batch) > 0:
            inputs = batch[0]
            if isinstance(inputs, dict):
                return self.model(**inputs)
            elif torch.is_tensor(inputs):
                return self.model(input_ids=inputs)
            else:
                raise TypeError(f"Unsupported input type: {type(inputs)}")
                
        elif torch.is_tensor(batch):
            return self.model(input_ids=batch)
        
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
    #

    def loss(self, outputs, batch):
        """Enhanced loss function with detailed error handling"""
        # Extract logits and loss from outputs (handle both tuple and object formats)
        if isinstance(outputs, tuple):
            # Debug the tuple structure
            print(f"DEBUG: outputs is tuple with {len(outputs)} elements")
            for i, item in enumerate(outputs[:3]):  # Print first 3 items
                print(f"DEBUG: outputs[{i}] type: {type(item)}, shape: {getattr(item, 'shape', 'No shape')}")
                
            # If tuple has 3 elements, it's likely (loss, logits, hidden_states)
            if len(outputs) == 3:
                loss = outputs[0] if outputs[0].numel() > 0 else None
                logits = outputs[1]  # Get logits from second position
            # If tuple has 2 elements, it's likely (logits, hidden_states)
            else:
                loss = None
                logits = outputs[0]
        elif hasattr(outputs, 'logits'):
            # Object format with attributes
            logits = outputs.logits
            loss = outputs.loss if hasattr(outputs, 'loss') else None
        elif isinstance(outputs, dict):
            # Dictionary format
            logits = outputs.get('logits')
            loss = outputs.get('loss')
        else:
            raise TypeError(f"Unsupported outputs type: {type(outputs)}")
        
        # Verify logits were extracted
        if logits is None:
            raise ValueError(f"Could not extract logits from outputs: {outputs}")
            
        # For debugging
        print(f"DEBUG: Final logits type: {type(logits)}, shape: {logits.shape}, ndim: {logits.ndim}")
        
        # Calculate loss if needed
        if loss is None or (hasattr(loss, 'numel') and loss.numel() == 0):  # Handle empty loss tensor case
            if logits.ndim == 0:  # Check if logits is a scalar
                print("WARNING: logits has no dimensions, creating dummy loss")
                # Return a dummy loss for training to continue
                return torch.tensor(1.0, requires_grad=True, device=logits.device)
            
            # Extract labels
            if isinstance(batch, dict) and "labels" in batch:
                labels = batch["labels"]
            elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                _, labels = batch
            else:
                print("WARNING: Cannot find labels in batch, creating dummy loss")
                return torch.tensor(1.0, requires_grad=True, device=logits.device)
            
            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss()
            try:
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
            except Exception as e:
                print(f"ERROR computing loss: {e}")
                print(f"logits shape: {logits.shape}, labels shape: {labels.shape}")
                # Fall back to dummy loss
                loss = torch.tensor(1.0, requires_grad=True, device=logits.device)
        
        return loss
    
        
    def get_state_dict(self, state_dict=None):
        """Custom state_dict handling that properly extracts PEFT adapter weights"""
        # Get the regular state_dict from parent method using the HuggingFace model
        regular_state_dict = self.model.get_state_dict(state_dict)
        
        if not self.using_peft:
            return regular_state_dict
            
        print("\nPreparing PEFT adapter state_dict for saving...")
        if hasattr(self.model, 'get_adapter_state_dict'):
            # Use PEFT's built-in function if available
            adapter_state_dict = self.model.get_adapter_state_dict()
        else:
            # Fall back to custom extraction
            adapter_state_dict = get_peft_model_state_dict(self.model)
            
        # Check if we have adapter weights
        if not adapter_state_dict:
            print("WARNING: No adapter weights found!")
        else:
            print(f"Found {len(adapter_state_dict)} adapter weights to save")
            # Save adapter separately for easier loading
            sample_key = list(adapter_state_dict.keys())[0] if adapter_state_dict else "No keys found"
            print(f"Sample adapter key: {sample_key}")
        
        # Save adapter_config and adapter_model separately
        save_folder = os.environ.get('COMPOSER_SAVE_FOLDER', '')
        if save_folder:
            save_path = Path(save_folder)
            parent_folder = save_path.parent.parent
            
            # Save config
            config_dict = {
                "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 8,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "inference_mode": False
            }
            
            config_path = parent_folder / "adapter_config.json"
            try:
                import json
                with open(config_path, "w") as f:
                    json.dump(config_dict, f, indent=2)
                print(f"Saved adapter config to {config_path}")
            except Exception as e:
                print(f"Error saving adapter config: {e}")
                
            # Save adapter model
            adapter_path = parent_folder / "adapter_model.bin"
            try:
                torch.save(adapter_state_dict, adapter_path)
                print(f"Saved adapter weights to {adapter_path}")
            except Exception as e:
                print(f"Error saving adapter weights: {e}")
        
        # For composer checkpoint, include adapter weights
        if self.should_save_peft_only:
            # Return only adapter weights
            print("Saving PEFT-only weights to Composer checkpoint")
            return {"model": adapter_state_dict}
        else:
            # Return full model with adapter weights
            print("Saving full model with adapter weights to Composer checkpoint")
            regular_state_dict["model"].update(adapter_state_dict)
            return regular_state_dict


# Now define the initialization function
def initialize_adapter():
    """Initialize the adapter by patching needed methods and registering the custom model"""
    # Add the missing method from HuggingFace's implementation to our custom implementation
    if not hasattr(LlamaForCausalLM, 'prepare_inputs_for_generation'):
        print("Adding prepare_inputs_for_generation method to our custom LlamaForCausalLM")
        prepare_inputs_for_generation = HFLlamaForCausalLM.prepare_inputs_for_generation
        setattr(LlamaForCausalLM, 'prepare_inputs_for_generation', prepare_inputs_for_generation)
        print("Successfully added method")

    # Register our model with the registry
    try:
        # CHANGE THIS LINE: Register as "hf_causal_lm" to match YAML
        registry.models.register("hf_causal_lm")(CustomLlamaModel)
        print("Successfully registered CustomLlamaModel as 'hf_causal_lm'")
        return True
    except Exception as e:
        print(f"Error registering CustomLlamaModel: {e}")
        import traceback
        traceback.print_exc()
        return False


# Actually initialize the adapter when this module is imported
initialize_adapter()




































# import sys
# import os
# import torch
# from composer.models import HuggingFaceModel, ComposerModel
# from llmfoundry.models.llama import LlamaForCausalLM
# from llmfoundry import registry
# from transformers import  AutoConfig, LlamaForCausalLM as HFLlamaForCausalLM
# from peft import LoraConfig, get_peft_model_state_dict
# from pathlib import Path

# # Add paths to Python path
# sys.path.insert(0, '/llm-foundry')
# sys.path.insert(0, '/llm-foundry/scripts')

# def initialize_adapter():
#     """Initialize the adapter by patching needed methods and registering the custom model"""
#     # Add the missing method from HuggingFace's implementation to our custom implementation
#     if not hasattr(LlamaForCausalLM, 'prepare_inputs_for_generation'):
#         print("Adding prepare_inputs_for_generation method to our custom LlamaForCausalLM")
#         prepare_inputs_for_generation = HFLlamaForCausalLM.prepare_inputs_for_generation
#         LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation
#         print("Successfully added method")

#     # Create our custom class that uses LlamaForCausalLM
#     try:
#         from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
        
#         # Create a proper subclass with exact signature match
#         class CustomLlamaModel(ComposerHFCausalLM):
#             def __init__(self, 
#                         pretrained_model_name_or_path,
#                         tokenizer=None,
#                         config=None,
#                         use_logits=False,
#                         metrics=None,
#                         eval_metrics=None,
#                         shift_labels=None,
#                         allow_embedding_resizing=False,
#                         peft_config=None,
#                         should_save_peft_only=True,
#                         init_device='meta',
#                         **kwargs):
#                 print("Initializing CustomLlamaModel")
                
#                 # Handle import_path and any other unwanted kwargs
#                 if 'import_path' in kwargs:
#                     del kwargs['import_path']
                
#                 # Make sure we're using a string for model path
#                 if not isinstance(pretrained_model_name_or_path, str):
#                     print(f"WARNING: pretrained_model_name_or_path is not a string: {type(pretrained_model_name_or_path)}")
#                     if hasattr(pretrained_model_name_or_path, 'name_or_path'):
#                         pretrained_model_name_or_path = pretrained_model_name_or_path.name_or_path
#                     else:
#                         pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B"
                
#                 print(f"Loading model from: {pretrained_model_name_or_path}")
                
#                 # Load configuration separately
#                 print("Loading model config...")
#                 if config is None:
#                     config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
#                     print(f"Loaded config: {type(config)}")
                
#                 # Use our custom LlamaForCausalLM implementation
#                 model = LlamaForCausalLM.from_pretrained(
#                     pretrained_model_name_or_path, 
#                     torch_dtype=torch.bfloat16,
#                     config=config,
#                     **kwargs
#                 )
                
#                 print(f"Using custom LlamaForCausalLM: {type(model).__name__}")
                
#                 # Ensure model has a config attribute
#                 if not hasattr(model, 'config'):
#                     print("Adding config attribute to model")
#                     model.config = config
                
#                 # Flag for PEFT tracking
#                 self.using_peft = peft_config is not None
#                 self.should_save_peft_only = should_save_peft_only
                
#                 # Convert peft_config dict to proper PEFT config object
#                 if peft_config is not None and isinstance(peft_config, dict):
#                     print(f"Converting peft_config dict to proper PEFT config: {peft_config}")
                    
#                     # Handle terminology differences
#                     if 'peft_type' not in peft_config and 'peft_method' in peft_config:
#                         peft_config['peft_type'] = peft_config['peft_method']
#                         del peft_config['peft_method']
                    
#                     # Check if it's a LoRA config
#                     if peft_config.get('peft_type', '').upper() == 'LORA':
#                         # Create proper LoraConfig
#                         lora_config = {
#                             'r': peft_config.get('r', 8),
#                             'lora_alpha': peft_config.get('lora_alpha', 16),
#                             'lora_dropout': peft_config.get('lora_dropout', 0.05),
#                             'target_modules': peft_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
#                             'bias': 'none',
#                             'task_type': peft_config.get('task_type', 'CAUSAL_LM'),
#                             'base_model_name_or_path': pretrained_model_name_or_path
#                         }
#                         print(f"Creating LoraConfig with: {lora_config}")
#                         peft_config = LoraConfig(**lora_config)
#                     else:
#                         print(f"Warning: Unsupported PEFT type: {peft_config.get('peft_type')}")
                
#                 # Initialize ComposerModel base
#                 ComposerModel.__init__(self)
                
#                 # Create the HuggingFaceModel wrapper
#                 hf_model = HuggingFaceModel(
#                     model=model,
#                     tokenizer=tokenizer,
#                     use_logits=use_logits,
#                     metrics=metrics,
#                     eval_metrics=eval_metrics,
#                     shift_labels=shift_labels,
#                     allow_embedding_resizing=allow_embedding_resizing,
#                     peft_config=peft_config,
#                     should_save_peft_only=should_save_peft_only
#                 )
                
#                 # Store the original model specifically
#                 self.hf_model = hf_model
#                 self.model = hf_model.model
                
#                 # Track weights
#                 self.forward_count = 0
                
#                 # Copy all attributes and methods from hf_model to self
#                 for key, value in vars(hf_model).items():
#                     if key not in ['hf_model', 'model']:  # Avoid circular references
#                         setattr(self, key, value)
                
#                 for name in dir(hf_model):
#                     if not name.startswith('_'):
#                         attr = getattr(hf_model, name)
#                         if callable(attr) and not hasattr(self, name):
#                             setattr(self, name, attr)
                                
#                 print("CustomLlamaModel initialization complete")
            
#             # Implement a custom loss function that handles both dict and tuple outputs
#             def loss(self, outputs, batch):
#                 print(f"CustomLlamaModel.loss called with outputs type: {type(outputs)}")
                
#                 # First get the basic loss
#                 if isinstance(outputs, dict):
#                     if 'loss' in outputs:
#                         loss = outputs['loss']
#                     else:
#                         # Calculate loss from logits
#                         logits = outputs['logits']
#                         labels = batch['labels']
#                         loss_fct = torch.nn.CrossEntropyLoss()
#                         loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#                 elif isinstance(outputs, tuple):
#                     # Assuming the loss is the first element in the tuple if present
#                     if len(outputs) > 0 and outputs[0] is not None:
#                         loss = outputs[0]
#                     else:
#                         # Fallback loss calculation
#                         logits = outputs[0]
#                         labels = batch['labels'] if isinstance(batch, dict) else batch[1]
#                         loss_fct = torch.nn.CrossEntropyLoss()
#                         loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#                 else:
#                     # Final fallback for unexpected output format
#                     print(f"WARNING: Unexpected outputs type: {type(outputs)}, using default loss calculation")
#                     # Try to extract logits from the outputs, assuming it has an attribute or can be indexed
#                     try:
#                         if hasattr(outputs, 'logits'):
#                             logits = outputs.logits
#                         else:
#                             logits = outputs  # Assume outputs is the logits tensor directly
                        
#                         labels = batch['labels'] if isinstance(batch, dict) else batch[1]
#                         loss_fct = torch.nn.CrossEntropyLoss()
#                         loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#                     except Exception as e:
#                         print(f"Error calculating loss: {e}")
#                         # Return a dummy loss as last resort (zero loss will make training stop)
#                         loss = torch.tensor(0.0, requires_grad=True)
                
#                 # Monitor adapter weights periodically (rest of the function remains the same)...
                
#                 return loss
#             #
#             def get_state_dict(self, state_dict=None):
#                 """Custom state_dict handling that properly extracts PEFT adapter weights"""
#                 # Get the regular state_dict from parent method
#                 regular_state_dict = self.hf_model.get_state_dict(state_dict)
                
#                 if not hasattr(self, 'using_peft') or not self.using_peft:
#                     return regular_state_dict
                    
#                 print("\nPreparing PEFT adapter state_dict for saving...")
#                 try:
#                     # Get adapter weights
#                     from peft import get_peft_model_state_dict
#                     adapter_state_dict = get_peft_model_state_dict(self.model)
                    
#                     # Check if we have adapter weights
#                     if not adapter_state_dict:
#                         print("WARNING: No adapter weights found!")
#                     else:
#                         print(f"Found {len(adapter_state_dict)} adapter weights to save")
#                         sample_key = list(adapter_state_dict.keys())[0] if adapter_state_dict else "No keys found"
#                         print(f"Sample adapter key: {sample_key}")
                        
#                         # For composer checkpoint, include adapter weights based on flag
#                         if hasattr(self, 'should_save_peft_only') and self.should_save_peft_only:
#                             print("Saving PEFT-only weights to Composer checkpoint")
#                             return {"model": adapter_state_dict}
#                         else:
#                             print("Saving full model with adapter weights to Composer checkpoint")
#                             regular_state_dict["model"].update(adapter_state_dict)
#                 except Exception as e:
#                     print(f"Error in get_state_dict: {e}")
#                     import traceback
#                     traceback.print_exc()
                    
#                 return regular_state_dict

#             def on_save_checkpoint(self, checkpoint_path):
#                 """Method called explicitly during checkpoint saving to extract adapter weights"""
#                 print(f"\non_save_checkpoint called with path: {checkpoint_path}")
                
#                 if not hasattr(self, 'using_peft') or not self.using_peft:
#                     print("Not a PEFT model, skipping adapter saving")
#                     return
                    
#                 print("Extracting adapter weights...")
#                 try:
#                     # Get adapter weights
#                     from peft import get_peft_model_state_dict
#                     adapter_state_dict = get_peft_model_state_dict(self.model)
                    
#                     if not adapter_state_dict:
#                         print("No adapter weights found!")
#                         return
                        
#                     print(f"Found {len(adapter_state_dict)} adapter weights")
                    
#                     # Save adapter files
#                     import os, json, torch
#                     from pathlib import Path
                    
#                     # Determine save location - primary path is two levels up from checkpoint
#                     checkpoint_dir = Path(checkpoint_path).parent
#                     run_dir = checkpoint_dir.parent.parent
                    
#                     # Create adapter config
#                     adapter_config = {
#                         "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
#                         "peft_type": "LORA",
#                         "task_type": "CAUSAL_LM",
#                         "r": 8,
#                         "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
#                         "lora_alpha": 16,
#                         "lora_dropout": 0.05,
#                         "inference_mode": False
#                     }
                    
#                     # Save files in run directory (standard location)
#                     adapter_config_path = run_dir / "adapter_config.json"
#                     adapter_weights_path = run_dir / "adapter_model.bin"
                    
#                     with open(adapter_config_path, "w") as f:
#                         json.dump(adapter_config, f, indent=2)
                        
#                     torch.save(adapter_state_dict, adapter_weights_path)
                    
#                     print(f"Saved adapter files to {run_dir}")
#                     print(f"- Config: {adapter_config_path}")
#                     print(f"- Weights: {adapter_weights_path}")
                    
#                     # Also save in checkpoint directory for backward compatibility
#                     checkpoint_config_path = checkpoint_dir / "adapter_config.json"
#                     checkpoint_weights_path = checkpoint_dir / "adapter_model.bin"
                    
#                     with open(checkpoint_config_path, "w") as f:
#                         json.dump(adapter_config, f, indent=2)
                        
#                     torch.save(adapter_state_dict, checkpoint_weights_path)
#                     print(f"Also saved adapter files in checkpoint directory: {checkpoint_dir}")
                    
#                 except Exception as e:
#                     print(f"Error saving adapter files: {e}")
#                     import traceback
#                     traceback.print_exc()

#         registry.models.register('hf_causal_lm', func=CustomLlamaModel)
#         print("CustomLlamaModel successfully registered as hf_causal_lm")
    
#             # # Add custom state_dict handler for adapter saving
#             # def get_state_dict(self, state_dict=None):
#             #     """Custom state_dict handling that properly extracts PEFT adapter weights"""
#             #     # Get the regular state_dict from parent method
#             #     regular_state_dict = self.hf_model.get_state_dict(state_dict)
                
#             #     if not hasattr(self, 'using_peft') or not self.using_peft:
#             #         return regular_state_dict
                    
#             #     print("\nPreparing PEFT adapter state_dict for saving...")
#             #     if hasattr(self.model, 'get_adapter_state_dict'):
#             #         # Use PEFT's built-in function if available
#             #         adapter_state_dict = self.model.get_adapter_state_dict()
#             #     else:
#             #         # Fall back to custom extraction
#             #         adapter_state_dict = get_peft_model_state_dict(self.model)
                    
#             #     # Check if we have adapter weights
#             #     if not adapter_state_dict:
#             #         print("WARNING: No adapter weights found!")
#             #     else:
#             #         print(f"Found {len(adapter_state_dict)} adapter weights to save")
#             #         # Save adapter separately for easier loading
#             #         sample_key = list(adapter_state_dict.keys())[0] if adapter_state_dict else "No keys found"
#             #         print(f"Sample adapter key: {sample_key}")
                
#             #     # Save adapter_config and adapter_model separately
#             #     save_folder = os.environ.get('COMPOSER_SAVE_FOLDER', '')
#             #     if save_folder:
#             #         save_path = Path(save_folder)
#             #         parent_folder = save_path.parent.parent
                    
#             #         # Save config
#             #         config_dict = {
#             #             "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
#             #             "peft_type": "LORA",
#             #             "task_type": "CAUSAL_LM",
#             #             "r": 8,
#             #             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
#             #             "lora_alpha": 16,
#             #             "lora_dropout": 0.05,
#             #             "inference_mode": False
#             #         }
                    
#             #         config_path = parent_folder / "adapter_config.json"
#             #         try:
#             #             import json
#             #             with open(config_path, "w") as f:
#             #                 json.dump(config_dict, f, indent=2)
#             #             print(f"Saved adapter config to {config_path}")
#             #         except Exception as e:
#             #             print(f"Error saving adapter config: {e}")
                        
#             #         # Save adapter model
#             #         adapter_path = parent_folder / "adapter_model.bin"
#             #         try:
#             #             torch.save(adapter_state_dict, adapter_path)
#             #             print(f"Saved adapter weights to {adapter_path}")
#             #         except Exception as e:
#             #             print(f"Error saving adapter weights: {e}")
                
#             #     # For composer checkpoint, include adapter weights
#             #     if hasattr(self, 'should_save_peft_only') and self.should_save_peft_only:
#             #         # Return only adapter weights
#             #         print("Saving PEFT-only weights to Composer checkpoint")
#             #         return {"model": adapter_state_dict}
#             #     else:
#             #         # Return full model with adapter weights
#             #         print("Saving full model with adapter weights to Composer checkpoint")
#             #         regular_state_dict["model"].update(adapter_state_dict)
#             #         return regular_state_dict
#             # def on_save_checkpoint(self, checkpoint_path):
#             #     """Hook explicitly called when saving checkpoints"""
#             #     print(f"\nSaving checkpoint to {checkpoint_path}")
                
#             #     if not hasattr(self, 'using_peft') or not self.using_peft:
#             #         print("Not a PEFT model, skipping adapter saving")
#             #         return
                
#             #     print("Extracting adapter weights...")
#             #     adapter_state_dict = {}
#             #     try:
#             #         # Try multiple methods to extract adapter weights
#             #         if hasattr(self.model, 'get_adapter_state_dict'):
#             #             adapter_state_dict = self.model.get_adapter_state_dict()
#             #             print("Used model.get_adapter_state_dict()")
#             #         elif hasattr(self.model, 'peft_config'):
#             #             from peft import get_peft_model_state_dict
#             #             adapter_state_dict = get_peft_model_state_dict(self.model)
#             #             print("Used get_peft_model_state_dict()")
#             #         else:
#             #             # Manual extraction - look for all LoRA parameters
#             #             for name, param in self.model.named_parameters():
#             #                 if 'lora_' in name and param.requires_grad:
#             #                     adapter_state_dict[name] = param.data.cpu().clone()
#             #             print("Used manual LoRA parameter extraction")
#             #     except Exception as e:
#             #         print(f"Error extracting adapter weights: {e}")
                
#             #     # Check if we found any adapter weights
#             #     if not adapter_state_dict:
#             #         print("WARNING: No adapter weights found! The model may not have PEFT/LoRA initialized properly.")
#             #         return
                
#             #     print(f"Found {len(adapter_state_dict)} adapter weights")
                
#             #     # Save adapter files next to checkpoint
#             #     checkpoint_dir = os.path.dirname(checkpoint_path)
#             #     parent_dir = os.path.dirname(checkpoint_dir)
                
#             #     # Create adapter config
#             #     config_dict = {
#             #         "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
#             #         "peft_type": "LORA",
#             #         "task_type": "CAUSAL_LM",
#             #         "r": 8,
#             #         "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
#             #         "lora_alpha": 16,
#             #         "lora_dropout": 0.05,
#             #         "inference_mode": False
#             #     }
                
#             #     # Save files in both locations to be safe
#             #     for save_dir in [checkpoint_dir, parent_dir]:
#             #         try:
#             #             # Save adapter config
#             #             config_path = os.path.join(save_dir, "adapter_config.json")
#             #             with open(config_path, "w") as f:
#             #                 import json
#             #                 json.dump(config_dict, f, indent=2)
                        
#             #             # Save adapter weights
#             #             adapter_path = os.path.join(save_dir, "adapter_model.bin")
#             #             torch.save(adapter_state_dict, adapter_path)
                        
#             #             print(f"Adapter files saved to {save_dir}")
#             #         except Exception as e:
#             #             print(f"Error saving adapter files to {save_dir}: {e}")
#             #         # Register our class
#             #         registry.models.register("hf_causal_lm")(CustomLlamaModel)
#             #         print("Successfully registered CustomLlamaModel as 'hf_causal_lm'")
#             #         return True
#     except Exception as e:
#         print(f"Error registering CustomLlamaModel: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# # Initialize adapter when the module is imported
# initialize_adapter()
