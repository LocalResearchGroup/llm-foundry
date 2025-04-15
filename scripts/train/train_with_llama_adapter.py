import sys
import os
from pathlib import Path

# Add paths to Python path - use relative paths instead of hardcoded ones
current_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(current_dir))

# Import our adapter
print("Importing llama adapter...")
try:
    from llmfoundry.models.llama import model_hf_register # Auto-initialize adapter upon import!
    print("Adapter imported successfully")
except ImportError as e:
    print(f"Error importing llama adapter: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Available modules in llmfoundry.models: {os.listdir(os.path.join(current_dir, 'llmfoundry', 'models'))}")
    raise

# Import necessary modules for model creation
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmfoundry.models.llama import LlamaForCausalLM

# Run the training function
from llmfoundry.command_utils import train_from_yaml

# Parse command line arguments - keep the same structure as before
if len(sys.argv) < 3:
    print("Usage: python train_with_adapter.py <yaml_path> <data_path>")
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

# Create a model instance to pass to the LlamaForCausalLM constructor
print("Creating base model instance...")
model_name = "meta-llama/Llama-3.2-1B"  # This will be overridden by the YAML config
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Register the model with the model parameter
print("Registering model with custom parameters...")
from llmfoundry.utils.registry_utils import registry
registry.models.register("hf_causal_lm")(lambda **kwargs: LlamaForCausalLM(model=base_model, **kwargs))

# Call train_from_yaml with all arguments
print("Starting training...")
print(f"Full arguments list: {args_list}")
train_from_yaml(yaml_path, args_list)