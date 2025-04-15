import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

import torch
from composer.models import HuggingFaceModel
from llmfoundry.models.llama import LlamaForCausalLM
from llmfoundry import registry
from llmfoundry.command_utils import train_from_yaml


# Add paths to Python path - use relative paths instead of hardcoded ones
current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_dir))


class CustomLlamaModel(HuggingFaceModel):
    """Custom Llama model that extends HuggingFaceModel."""
    
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer: Optional[Any] = None,
        use_flash_attention_2: bool = True,
        peft_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the custom Llama model.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained model
            tokenizer: Tokenizer to use
            use_flash_attention_2: Whether to use Flash Attention 2
            peft_config: Optional PEFT configuration
            **kwargs: Additional arguments to pass to model
        """
        # Remove any parameters that might cause issues
        if 'import_path' in kwargs:
            del kwargs['import_path']
        
        # Load the model using our custom implementation
        model = LlamaForCausalLM.from_pretrained(
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
    
    def forward(self, batch: Dict[str, Any]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Custom forward method to handle the model outputs correctly.
        
        Args:
            batch: Input batch containing input_ids and labels
            
        Returns:
            Model outputs as a dictionary with 'loss' and 'logits' keys
        """
        # Extract input_ids and labels from batch
        input_ids = batch['input_ids']
        labels = batch.get('labels', None)
        
        # Forward pass through the model
        outputs = self.model(input_ids=input_ids, labels=labels)
        
        # Convert tuple outputs to dictionary format
        if isinstance(outputs, tuple):
            loss = outputs[0]
            logits = outputs[1]
        else:
            loss = outputs.loss
            logits = outputs.logits
        
        return {
            'loss': loss,
            'logits': logits
        }
    
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


# Register our custom model class
registry.models.register("hf_causal_lm.custom_llama")(CustomLlamaModel)
print("Registered CustomLlamaModel as hf_causal_lm.custom_llama")


# Parse command line arguments
if len(sys.argv) < 3:
    print("Usage: python train_with_custom_llama.py <yaml_path> <data_path>")
    sys.exit(1)

yaml_path = sys.argv[1]
data_path = sys.argv[2]

print(f"Running training with YAML: {yaml_path}")
print(f"Data path: {data_path}")

# Set up args list with data path
args_list = [f"variables.data_local={data_path}"]

# Add any additional CLI arguments
if len(sys.argv) > 3:
    additional_args = sys.argv[3:]
    print(f"Received {len(additional_args)} additional arguments")
    args_list.extend(additional_args)

# Call train_from_yaml with all arguments
print("Starting training...")
print(f"Full arguments list: {args_list}")
train_from_yaml(yaml_path, args_list) 