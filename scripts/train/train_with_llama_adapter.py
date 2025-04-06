import sys
import os

# Add paths to Python path
sys.path.insert(0, '/llm-foundry')

# Import our adapter
print("Importing llama adapter...")
from llmfoundry.models.llama import composer_llama_adapter # Auto-initialize adapter upon import!
print("Adapter imported successfully")

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

# Call train_from_yaml with the data path
print("Starting training...")
args_list = [f"variables.data_local={data_path}"]
train_from_yaml(yaml_path, args_list)