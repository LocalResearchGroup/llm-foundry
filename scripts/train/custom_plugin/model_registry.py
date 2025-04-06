import torch
from composer.models import HuggingFaceModel
from llmfoundry.models.llama import LlamaForCausalLM
#from llmfoundry.utils.registry_utils import construct_from_registry
from llmfoundry import registry


# Define our model builder function
def build_llama3_1b(
    pretrained_model_name_or_path,
    tokenizer=None,
    **kwargs
):
    print("Building model using CUSTOM LlamaForCausalLM implementation")
    
    # Remove any parameters that might cause issues
    if 'import_path' in kwargs:
        del kwargs['import_path']
        
    # Load the model using our custom implementation
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype=torch.bfloat16,
        **kwargs
    )
    
    print(f"Model type: {type(model).__name__}")
    
    # Wrap with Composer's model class
    return HuggingFaceModel(
        model=model,
        tokenizer=tokenizer
    )

# Direct registration - this is the correct way when registry_module isn't available
registry.models.register("llama3_1b")(build_llama3_1b)
print("Registered llama3_1b with the registry")


# import torch
# import sys
# sys.path.append('/llm-foundry')

# # Import registry decorator
# from llmfoundry.utils.registry_utils import registry_module

# # Import your custom implementation
# from llmfoundry.models.llama import LlamaForCausalLM

# # Import Composer's model wrapper
# from composer.models import HuggingFaceModel

# # Add the required decorator here
# @registry_module(registry_name='models')
# def build_llama3_1b(
#     pretrained_model_name_or_path,
#     use_flash_attention_2=True,
#     peft_config=None,
#     **kwargs
# ):
#     """
#     Must have build_<model_name> since build_composer_model in composer's
#     train.py will use builder_fn = registry.get_module_builder(name) to 
#     build from import path behind the scenes.
#     """
#     print(f"Building model using custom LlamaForCausalLM from {pretrained_model_name_or_path}")
    
#     # Load the model using your custom implementation
#     model = LlamaForCausalLM.from_pretrained(
#         pretrained_model_name_or_path,
#         torch_dtype=torch.bfloat16,
#         **kwargs
#     )
    
#     # Apply PEFT if specified
#     if peft_config:
#         from peft import get_peft_model, LoraConfig
#         peft_type = peft_config.get('peft_type', 'LORA')
        
#         if peft_type == 'LORA':
#             lora_config = LoraConfig(
#                 r=peft_config.get('r', 8),
#                 lora_alpha=peft_config.get('lora_alpha', 16),
#                 lora_dropout=peft_config.get('lora_dropout', 0.05),
#                 target_modules=peft_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
#                 bias=peft_config.get('bias', 'none'),
#                 task_type=peft_config.get('task_type', 'CAUSAL_LM')
#             )
#             model = get_peft_model(model, lora_config)
    
#     # Wrap in Composer's model class
#     composer_model = HuggingFaceModel(
#         model=model,
#         tokenizer=None,  # Will be loaded separately
#         use_logits=True
#     )
    
#     return composer_model