"""Initializes HuggingFaceModel using custom architecture (LlamaForCausalLM) 
Loading weights into our custom model from the pretrained checkpoint
Mimicking the expected interface by creating a wrapper class that matches what 
Composer expects (ComposerHFCausalLM)
Instead of modifying the framework, we're making our custom implementation look 
like what the framework expects.
"""

import sys
import os
import torch
import inspect
from composer.models import HuggingFaceModel, ComposerModel
from llmfoundry.models.llama import LlamaForCausalLM
from llmfoundry import registry
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM as HFLlamaForCausalLM

# Add paths to Python path
sys.path.insert(0, '/llm-foundry')
sys.path.insert(0, '/llm-foundry/scripts')

def initialize_adapter():
    """Initialize the adapter by patching needed methods and registering the custom model"""
    # Add the missing method from HuggingFace's implementation to our custom implementation
    if not hasattr(LlamaForCausalLM, 'prepare_inputs_for_generation'):
        print("Adding prepare_inputs_for_generation method to our custom LlamaForCausalLM")
        prepare_inputs_for_generation = HFLlamaForCausalLM.prepare_inputs_for_generation
        LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation
        print("Successfully added method")

    # Create our custom class that uses LlamaForCausalLM
    try:
        from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
        from peft import LoraConfig
        
        # Create a proper subclass with exact signature match
        class CustomLlamaModel(ComposerHFCausalLM):
            def __init__(self, 
                        pretrained_model_name_or_path,
                        tokenizer=None,
                        config=None,
                        use_logits=False,
                        metrics=None,
                        eval_metrics=None,
                        shift_labels=None,
                        allow_embedding_resizing=False,
                        peft_config=None,
                        should_save_peft_only=True,
                        init_device='meta',
                        **kwargs):
                print("Initializing CustomLlamaModel")
                
                # Handle import_path and any other unwanted kwargs
                if 'import_path' in kwargs:
                    del kwargs['import_path']
                
                # Make sure we're using a string for model path
                if not isinstance(pretrained_model_name_or_path, str):
                    print(f"WARNING: pretrained_model_name_or_path is not a string: {type(pretrained_model_name_or_path)}")
                    if hasattr(pretrained_model_name_or_path, 'name_or_path'):
                        pretrained_model_name_or_path = pretrained_model_name_or_path.name_or_path
                    else:
                        pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B"
                
                print(f"Loading model from: {pretrained_model_name_or_path}")
                
                # Load configuration separately
                print("Loading model config...")
                if config is None:
                    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
                    print(f"Loaded config: {type(config)}")
                
                # Use our custom LlamaForCausalLM implementation
                model = LlamaForCausalLM.from_pretrained(
                    pretrained_model_name_or_path, 
                    torch_dtype=torch.bfloat16,
                    config=config,
                    **kwargs
                )
                
                print(f"Using custom LlamaForCausalLM: {type(model).__name__}")
                
                # Ensure model has a config attribute
                if not hasattr(model, 'config'):
                    print("Adding config attribute to model")
                    model.config = config
                
                # Convert peft_config dict to proper PEFT config object
                if peft_config is not None and isinstance(peft_config, dict):
                    print(f"Converting peft_config dict to proper PEFT config: {peft_config}")
                    
                    # Handle terminology differences
                    if 'peft_type' not in peft_config and 'peft_method' in peft_config:
                        peft_config['peft_type'] = peft_config['peft_method']
                        del peft_config['peft_method']
                    
                    # Check if it's a LoRA config
                    if peft_config.get('peft_type', '').upper() == 'LORA':
                        # Create proper LoraConfig
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
                    else:
                        print(f"Warning: Unsupported PEFT type: {peft_config.get('peft_type')}")
                
                # Initialize ComposerModel base
                ComposerModel.__init__(self)
                
                # Create the HuggingFaceModel wrapper
                hf_model = HuggingFaceModel(
                    model=model,
                    tokenizer=tokenizer,
                    use_logits=use_logits,
                    metrics=metrics,
                    eval_metrics=eval_metrics,
                    shift_labels=shift_labels,
                    allow_embedding_resizing=allow_embedding_resizing,
                    peft_config=peft_config,
                    should_save_peft_only=should_save_peft_only
                )
                
                # Store the original model specifically
                self.hf_model = hf_model
                self.model = hf_model.model
                
                # Copy all attributes and methods from hf_model to self
                for key, value in vars(hf_model).items():
                    if key not in ['hf_model', 'model']:  # Avoid circular references
                        setattr(self, key, value)
                
                for name in dir(hf_model):
                    if not name.startswith('_'):
                        attr = getattr(hf_model, name)
                        if callable(attr) and not hasattr(self, name):
                            setattr(self, name, attr)
                                
                print("CustomLlamaModel initialization complete")
            
            # Implement a custom loss function that handles both dict and tuple outputs
            def loss(self, outputs, batch):
                print(f"CustomLlamaModel.loss called with outputs type: {type(outputs)}")
                
                if isinstance(outputs, dict):
                    if 'loss' in outputs:
                        return outputs['loss']
                elif isinstance(outputs, tuple):
                    # Assuming the loss is the first element in the tuple if present
                    if len(outputs) > 0 and outputs[0] is not None:
                        return outputs[0]
                
                # If we can't find the loss in the outputs, try to compute it
                # This is a fallback and might not be needed if your model returns loss correctly
                print("Computing loss from logits and labels")
                
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
                labels = batch['labels'] if isinstance(batch, dict) else batch[1]
                
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss()
                vocab_size = getattr(self.model, 'vocab_size', shift_logits.size(-1))
                loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                
                return loss
        
        # Register our class
        registry.models.register("hf_causal_lm")(CustomLlamaModel)
        print("Registered CustomLlamaModel as hf_causal_lm")
        return True
    except Exception as e:
        print(f"Error creating CustomLlamaModel: {e}")
        import traceback
        traceback.print_exc()
        return False

# Initialize adapter when the module is imported
initialize_adapter()

# import sys
# import os
# import torch
# import inspect
# from composer.models import HuggingFaceModel, ComposerModel
# from llmfoundry.models.llama import LlamaForCausalLM
# from llmfoundry import registry
# from llmfoundry.command_utils import train_from_yaml
# from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM as HFLlamaForCausalLM

# # Add paths to Python path
# sys.path.insert(0, '/llm-foundry')
# sys.path.insert(0, '/llm-foundry/scripts')

# # Add the missing method from HuggingFace's implementation to our custom implementation
# if not hasattr(LlamaForCausalLM, 'prepare_inputs_for_generation'):
#     print("Adding prepare_inputs_for_generation method to our custom LlamaForCausalLM")
#     prepare_inputs_for_generation = HFLlamaForCausalLM.prepare_inputs_for_generation
#     LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation
#     print("Successfully added method")

# # Create our custom class that uses LlamaForCausalLM
# try:
#     from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
#     from peft import LoraConfig
    
#     # Create a proper subclass with exact signature match
#     class CustomLlamaModel(ComposerHFCausalLM):
#         def __init__(self, 
#                     pretrained_model_name_or_path,
#                     tokenizer=None,
#                     config=None,
#                     use_logits=False,
#                     metrics=None,
#                     eval_metrics=None,
#                     shift_labels=None,
#                     allow_embedding_resizing=False,
#                     peft_config=None,
#                     should_save_peft_only=True,
#                     init_device='meta',
#                     **kwargs):
#             print("Initializing CustomLlamaModel")
            
#             # Handle import_path and any other unwanted kwargs
#             if 'import_path' in kwargs:
#                 del kwargs['import_path']
            
#             # Make sure we're using a string for model path
#             if not isinstance(pretrained_model_name_or_path, str):
#                 print(f"WARNING: pretrained_model_name_or_path is not a string: {type(pretrained_model_name_or_path)}")
#                 if hasattr(pretrained_model_name_or_path, 'name_or_path'):
#                     pretrained_model_name_or_path = pretrained_model_name_or_path.name_or_path
#                 else:
#                     pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B"
            
#             print(f"Loading model from: {pretrained_model_name_or_path}")
            
#             # Load configuration separately
#             print("Loading model config...")
#             if config is None:
#                 config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
#                 print(f"Loaded config: {type(config)}")
            
#             # Use our custom LlamaForCausalLM implementation
#             model = LlamaForCausalLM.from_pretrained(
#                 pretrained_model_name_or_path, 
#                 torch_dtype=torch.bfloat16,
#                 config=config,
#                 **kwargs
#             )
            
#             print(f"Using custom LlamaForCausalLM: {type(model).__name__}")
            
#             # Ensure model has a config attribute
#             if not hasattr(model, 'config'):
#                 print("Adding config attribute to model")
#                 model.config = config
            
#             # Convert peft_config dict to proper PEFT config object
#             if peft_config is not None and isinstance(peft_config, dict):
#                 print(f"Converting peft_config dict to proper PEFT config: {peft_config}")
                
#                 # Handle terminology differences
#                 if 'peft_type' not in peft_config and 'peft_method' in peft_config:
#                     peft_config['peft_type'] = peft_config['peft_method']
#                     del peft_config['peft_method']
                
#                 # Check if it's a LoRA config
#                 if peft_config.get('peft_type', '').upper() == 'LORA':
#                     # Create proper LoraConfig
#                     lora_config = {
#                         'r': peft_config.get('r', 8),
#                         'lora_alpha': peft_config.get('lora_alpha', 16),
#                         'lora_dropout': peft_config.get('lora_dropout', 0.05),
#                         'target_modules': peft_config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
#                         'bias': 'none',
#                         'task_type': peft_config.get('task_type', 'CAUSAL_LM'),
#                         'base_model_name_or_path': pretrained_model_name_or_path
#                     }
#                     print(f"Creating LoraConfig with: {lora_config}")
#                     peft_config = LoraConfig(**lora_config)
#                 else:
#                     print(f"Warning: Unsupported PEFT type: {peft_config.get('peft_type')}")
            
#             # Initialize ComposerModel base
#             ComposerModel.__init__(self)
            
#             # Create the HuggingFaceModel wrapper
#             hf_model = HuggingFaceModel(
#                 model=model,
#                 tokenizer=tokenizer,
#                 use_logits=use_logits,
#                 metrics=metrics,
#                 eval_metrics=eval_metrics,
#                 shift_labels=shift_labels,
#                 allow_embedding_resizing=allow_embedding_resizing,
#                 peft_config=peft_config,
#                 should_save_peft_only=should_save_peft_only
#             )
            
#             # Store the original model specifically
#             self.hf_model = hf_model
#             self.model = hf_model.model
            
#             # Copy all attributes and methods from hf_model to self
#             for key, value in vars(hf_model).items():
#                 if key not in ['hf_model', 'model']:  # Avoid circular references
#                     setattr(self, key, value)
            
#             for name in dir(hf_model):
#                 if not name.startswith('_'):
#                     attr = getattr(hf_model, name)
#                     if callable(attr) and not hasattr(self, name):
#                         setattr(self, name, attr)
                            
#             print("CustomLlamaModel initialization complete")
        
#         # Implement a custom loss function that handles both dict and tuple outputs
#         def loss(self, outputs, batch):
#             print(f"CustomLlamaModel.loss called with outputs type: {type(outputs)}")
            
#             if isinstance(outputs, dict):
#                 if 'loss' in outputs:
#                     return outputs['loss']
#             elif isinstance(outputs, tuple):
#                 # Assuming the loss is the first element in the tuple if present
#                 if len(outputs) > 0 and outputs[0] is not None:
#                     return outputs[0]
            
#             # If we can't find the loss in the outputs, try to compute it
#             # This is a fallback and might not be needed if your model returns loss correctly
#             print("Computing loss from logits and labels")
            
#             logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]
#             labels = batch['labels'] if isinstance(batch, dict) else batch[1]
            
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
            
#             # Flatten the tokens
#             loss_fct = torch.nn.CrossEntropyLoss()
#             vocab_size = getattr(self.model, 'vocab_size', shift_logits.size(-1))
#             loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            
#             return loss
    
#     # Register our class
#     registry.models.register("hf_causal_lm")(CustomLlamaModel)
#     print("Registered CustomLlamaModel as hf_causal_lm")
# except Exception as e:
#     print(f"Error creating CustomLlamaModel: {e}")
#     import traceback
#     traceback.print_exc()

# if __name__ == "__main__":
#     # Execute only if run as a script
#     if len(sys.argv) < 3:
#         print("Usage: python train_adapter.py <yaml_path> <data_path>")
#         sys.exit(1)
        
#     yaml_path = sys.argv[1]
#     data_path = sys.argv[2]

#     print(f"Running training with YAML: {yaml_path}")
#     print(f"Data path: {data_path}")

#     # Call the train function directly in this process
#     print("Starting training...")
#     args_list = [f"variables.data_local={data_path}"]
#     train_from_yaml(yaml_path, args_list)