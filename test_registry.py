import os
import subprocess
python_path = "/opt/conda/envs/llm-foundry/bin/python"

def test_registry():
    """Create a test script to verify the registry"""
    os.makedirs("/llm-foundry/scripts/train/test_script", exist_ok=True)
    
    with open("/llm-foundry/scripts/train/test_script/test_registry.py", "w") as f:
        f.write("""
import sys
sys.path.append('/llm-foundry')

import importlib
print("Importing module...")
try:
    mod = importlib.import_module('scripts.train.custom_plugin.model_registry')
    print("Successfully imported module")
except Exception as e:
    print(f"Error importing module: {e}")

from llmfoundry import registry
print("Available models:", list(registry.models.get_all().keys()))
print("Looking for llama3_1b...")
try:
    builder = registry.models.get("llama3_1b")
    print(f"Found llama3_1b: {builder}")
except Exception as e:
    print(f"Error getting llama3_1b: {e}")
""")
    
    return "/llm-foundry/scripts/train/test_script/test_registry.py"

# Run the test script
test_script = test_registry()
os.chdir("/llm-foundry/scripts")
print("\nTesting registry...")
test_result = subprocess.run([python_path, test_script], capture_output=True, text=True)
print(test_result.stdout)
if test_result.stderr:
    print(f"Test errors: {test_result.stderr}")