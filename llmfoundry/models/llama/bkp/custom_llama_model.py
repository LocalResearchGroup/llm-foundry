# custom_llama_model.py - A single file to handle everything

import torch
from composer.models import HuggingFaceModel
from pathlib import Path
import os
import json

# Import your custom model implementation
from llmfoundry.models.llama.bkp.model import LlamaForCausalLM

def create_llama_composer_model(
    pretrained_model_name_or_path,
    tokenizer,
    peft_config=None,
    use_flash_attention_2=False,
    trust_remote_code=True,
    **kwargs
):
    """Create a Composer-compatible LlamaForCausalLM model with adapter support.
    
    This function:
    1. Instantiates your custom LlamaForCausalLM implementation
    2. Wraps it in Composer's HuggingFaceModel
    3. Returns a model ready for training with adapter support
    """
    print(f"Creating LlamaForCausalLM from {pretrained_model_name_or_path}")
    
    # Create the base model using your custom implementation
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
    }
    
    if use_flash_attention_2:
        model_kwargs["use_flash_attention_2"] = True
    
    # Instantiate your custom model
    llama_model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        **model_kwargs
    )
    
    print(f"Created model: {type(llama_model).__name__}")
    
    # Wrap it with Composer's HuggingFaceModel
    composer_model = HuggingFaceModel(
        model=llama_model,
        tokenizer=tokenizer,
        use_logits=False,
        shift_labels=True,  # For causal LM
        peft_config=peft_config,
        should_save_peft_only=True,  # Save only adapter weights
        **kwargs
    )
    
    # Add an on_save_checkpoint hook for adapter extraction
    original_on_save = getattr(composer_model, 'on_save_checkpoint', None)
    
    def on_save_checkpoint(self, checkpoint_path):
        """Extract adapter weights to standard format during checkpointing"""
        print(f"Saving checkpoint to {checkpoint_path}")
        
        # Call original method if it exists
        if original_on_save:
            original_on_save(checkpoint_path)
        
        # Only extract adapters if using PEFT
        if not hasattr(self.model, 'peft_config'):
            print("Not using PEFT, skipping adapter extraction")
            return
            
        try:
            from peft import get_peft_model_state_dict
            
            # Extract adapter weights
            adapter_state_dict = get_peft_model_state_dict(self.model)
            
            # Create adapter files in parent directory
            checkpoint_dir = Path(checkpoint_path).parent
            output_dir = checkpoint_dir.parent
            
            config_path = output_dir / "adapter_config.json"
            weights_path = output_dir / "adapter_model.bin"
            
            # Create adapter config
            config_dict = {
                "base_model_name_or_path": pretrained_model_name_or_path,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": peft_config.get("r", 8),
                "target_modules": peft_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                "lora_alpha": peft_config.get("lora_alpha", 16),
                "lora_dropout": peft_config.get("lora_dropout", 0.05),
                "inference_mode": False
            }
            
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
                
            torch.save(adapter_state_dict, weights_path)
            
            print(f"Saved adapter files to {output_dir}")
            print(f"  - Config: {config_path}")
            print(f"  - Weights: {weights_path}")
            
        except Exception as e:
            print(f"Error extracting adapter: {e}")
            import traceback
            traceback.print_exc()
    
    # Attach the method to the instance
    composer_model.on_save_checkpoint = on_save_checkpoint.__get__(composer_model)
    
    return composer_model