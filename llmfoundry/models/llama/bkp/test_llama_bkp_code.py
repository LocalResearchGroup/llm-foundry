




## making sure actual adaptors are saved
@app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              max_containers=1)
def compare_generations(checkpoint_path: str):
    """Compare responses from base model vs adapter model"""
    import subprocess, os
    from pathlib import Path
    import json
    setup_hf_auth()
    
    model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
    
    print(f"Comparing base model vs adapter model from {model_path}")
    
    # Test prompts
    prompts = [
        "The answer to life, the universe, and happiness is",
        "Here's a quick recipe for baking chocolate chip cookies: Start by"
    ]
    
    compare_cmd = [
        PYTHON_PATH, "-c", f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
import json, os
set_seed(42)
model_path = "{model_path}"
prompts = {json.dumps(prompts)}

# First, check adapter_model.bin content
adapter_bin = os.path.join(model_path, "adapter_model.bin")
if os.path.exists(adapter_bin):
    adapter_dict = torch.load(adapter_bin, map_location="cpu")
    print(f"\\nAdapter model size: {{len(adapter_dict)}}")
    print(f"Sample keys: {{list(adapter_dict.keys())[:5] if adapter_dict else 'Empty'}}")
    
# Load tokenizer
print("\\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Load base model without adapter
print("\\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate with base model
print("\\n=== BASE MODEL RESPONSES ===")
base_responses = []
for prompt in prompts:
    print(f"\\nPrompt: {{prompt}}")
    inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
    with torch.no_grad():
        outputs = base_model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            #seed=42  # Fixed seed for consistent comparison
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = response[len(prompt):]
    print(f"Base Response: {{completion}}")
    base_responses.append(completion)

# Now try with adapter
print("\\n=== ADAPTER MODEL RESPONSES ===")
try:
    print(f"Loading adapter from {{model_path}}...")
    adapter_model = PeftModel.from_pretrained(base_model, model_path)
    
    adapter_responses = []
    for i, prompt in enumerate(prompts):
        print(f"\\nPrompt: {{prompt}}")
        inputs = tokenizer(prompt, return_tensors="pt").to(adapter_model.device)
        with torch.no_grad():
            outputs = adapter_model.generate(
                inputs.input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                #seed=42  # Fixed seed for consistent comparison
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = response[len(prompt):]
        print(f"Adapter Response: {{completion}}")
        adapter_responses.append(completion)
        
        # Compare responses
        if base_responses[i] == completion:
            print("⚠️ SAME OUTPUT: Adapter not affecting generation!")
        else:
            print("✅ DIFFERENT OUTPUT: Adapter is working")
    
except Exception as e:
    print(f"Error loading or using adapter: {{e}}")

"""
    ]
    
    subprocess.run(compare_cmd)
    return "Comparison complete"

@app.local_entrypoint()
def compare_existing_model():
    # Replace with your actual checkpoint path
    
    existing_model_path = "/model-checkpoints/llama3-1b-lora2-20250410_184857/native_checkpoints/latest-rank0.pt"
    
    # Run the comparison on the existing model
    compare_generations.remote(existing_model_path)
    
    return "Comparison completed!"

#############
    
def ensure_tokenizer_path(model_dir):
    """Ensure tokenizer has proper name_or_path"""
    import json
    import os
 
    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, 'r') as f:
            config = json.load(f)
        
        if "name_or_path" not in config or not config["name_or_path"]:
            print("Fixing empty tokenizer name_or_path...")
            config["name_or_path"] = "meta-llama/Llama-3.2-1B"
            
            with open(tokenizer_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print("Tokenizer config updated with correct name_or_path")



# def train_model(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml",
#                 hf_token: str = ''):
#     import os, subprocess, shutil, glob, time
#     from pathlib import Path
#     from omegaconf import OmegaConf
    
#     # Change to llm-foundry/scripts directory at the start
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     # Step 2: Train the model
#     print("\nTraining model...")
    
#     # Load YAML config and safely access variables
#     yaml_config = OmegaConf.load(yaml_path)
    
#     # Check if the config has a model_output_path variable
#     base_output_path = None
#     if 'variables' in yaml_config and 'model_output_path' in yaml_config.variables:
#         base_output_path = yaml_config.variables.model_output_path
#         print(f"Found model_output_path in YAML: {base_output_path}")
    
#     # Get model name and run folder
#     model_name = get_model_name(yaml_path)
#     run_folder = get_run_folder(run_ts, model_name)
    
#     # If base_output_path was found in YAML, use it to construct save_folder
#     if base_output_path:
#         # Strip trailing slash if present
#         base_output_path = base_output_path.rstrip('/')
#         # Create a path with timestamped folder under the base path
#         save_folder = Path(f"{base_output_path}/{model_name}-{run_ts}/native_checkpoints")
#     else:
#         # Fall back to default path construction
#         save_folder = Path(f"{run_folder}/native_checkpoints")
    
#     print(f"Using save folder: {save_folder}")
    
#     # Ensure directory exists
#     Path(save_folder).mkdir(exist_ok=True, parents=True)
#     shutil.copy(yaml_path, Path(save_folder) / Path(yaml_path).name)
    
#     # Use a consistent data path
#     data_path = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
    
#     # Setup HF token
#     if hf_token:
#         os.environ["HUGGINGFACE_TOKEN"] = hf_token
#         os.environ["HF_TOKEN"] = hf_token
#         os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
#         print("Using HF token passed from parent function")
        
#         login_cmd = ["huggingface-cli", "login", "--token", hf_token]
#         try:
#             subprocess.run(login_cmd, check=True, capture_output=True)
#             print("Logged in to HuggingFace using CLI")
#         except Exception as e:
#             print(f"Warning: HF CLI login failed: {e}")
    
#     # Setup training command based on selected mode
#     global USE_CUSTOM_MODEL
#     if USE_CUSTOM_MODEL:
#         train_cmd = [
#             PYTHON_PATH,
#             "train/train_with_llama_adapter.py",
#             yaml_path, 
#             data_path,
#             f"save_folder={save_folder}",
#             f"save_interval={SAVE_INTERVAL}",  # Save every batch to ensure checkpoints
#             "save_latest_filename=latest-rank0.pt",
#             f"max_duration={TRAIN_DURATION}", 
#             # Skip evaluation to speed up training
#             "eval_interval=0",
#             f"device_train_microbatch_size={BATCH_SIZE}",
#             f"global_train_batch_size={BATCH_SIZE}",
#         ]
#     else:
#         train_cmd = [
#             "composer",
#             "train/train.py",
#             yaml_path,
#             f"loggers.aim.experiment_name=quickstart_{model_name}_modal",
#             f"loggers.aim.repo={run_folder}/.aim",
#             f"variables.data_local={data_path}",
#             f"save_folder={save_folder}",
#             f"save_interval={SAVE_INTERVAL}",
#             f"max_duration={TRAIN_DURATION}",
#             "eval_interval=0", 
#             f"device_train_microbatch_size={BATCH_SIZE}",
#             f"global_train_batch_size={BATCH_SIZE}",
#         ]
    
#     # Run training, streaming output to console
#     print(f"Running command: {' '.join(train_cmd)}")
    
#     # CRITICAL CHANGE: Run without capturing output to allow streaming
#     result = subprocess.run(train_cmd)
    
#     # After training completes, check for checkpoints
#     print("\nChecking for checkpoint files...")
#     time.sleep(2)  # Small delay to ensure filesystem sync
    
#     pt_files = list(glob.glob(f"{save_folder}/**/*.pt", recursive=True))
#     if pt_files:
#         print(f"✅ Found {len(pt_files)} checkpoint files:")
#         for pt_file in pt_files:
#             file_size_mb = os.path.getsize(pt_file) / (1024 * 1024) 
#             print(f"  {pt_file} ({file_size_mb:.2f} MB)")
#     else:
#         print("⚠️ No checkpoint files found! Training may not have saved weights.")
    
#     # Save tokenizer explicitly right after training
#     print("\nSaving tokenizer to model directory...")
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#     tokenizer.save_pretrained(run_folder)
#     print(f"Tokenizer saved to {run_folder}")
    
#     MODEL_CHECKPOINT_VOLUME.commit()
#     print(f'Training complete for {run_ts}')
    
#     if result.returncode != 0:
#         raise Exception(f"Training failed with exit code {result.returncode}")
    
#     return str(run_folder)

# def train_model(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml",
#                 hf_token: str = ''):
#     import os, subprocess, shutil, time
#     from pathlib import Path
#     from omegaconf import OmegaConf
    
#     # Change to llm-foundry/scripts directory
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     # Load YAML config and set up paths
#     print("\nTraining model...")
#     yaml_config = OmegaConf.load(yaml_path)
    
#     # Get proper model paths
#     model_name = get_model_name(yaml_path)
#     run_folder = get_run_folder(run_ts, model_name)
#     save_folder = Path(f"{run_folder}/native_checkpoints")
#     save_folder.mkdir(exist_ok=True, parents=True)
    
#     # Copy YAML file to save folder
#     shutil.copy(yaml_path, save_folder / Path(yaml_path).name)
    
#     # Set HF token if needed
#     if hf_token:
#         os.environ["HUGGINGFACE_TOKEN"] = hf_token
#         os.environ["HF_TOKEN"] = hf_token
#         os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
#         print("Using HF token for training")
    
#     # Create a minimal, clean adapter script
#     minimal_adapter = """
# import sys
# from llmfoundry.command_utils import train_from_yaml
# from llmfoundry.models.llama import composer_llama_adapter

# # Print arguments for debugging
# print(f"Running with args: {sys.argv}")

# if __name__ == '__main__':
#     yaml_path, args_list = sys.argv[1], sys.argv[2:]
#     print(f"Starting training with YAML: {yaml_path}")
#     print(f"Args: {args_list}")
#     train_from_yaml(yaml_path, args_list)
#     print("Training completed successfully")
# """
    
#     # Write minimal adapter to a temporary file
#     minimal_adapter_path = "/tmp/minimal_train_adapter.py"
#     with open(minimal_adapter_path, "w") as f:
#         f.write(minimal_adapter)
    
#     # Prepare data path and arguments
#     data_path = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
    
#     # Build command with explicit arguments
#     train_cmd = [
#         PYTHON_PATH,
#         minimal_adapter_path,
#         yaml_path,
#         f"save_folder={save_folder}",
#         f"save_interval=1ba",  # Save every batch
#         f"save_latest_filename=latest-rank0.pt",
#         f"variables.data_local={data_path}",
#         f"max_duration=2ba",
#         f"eval_interval=0",  # No evaluation
#         f"device_train_microbatch_size={BATCH_SIZE}",
#         f"global_train_batch_size={BATCH_SIZE}"
#     ]
    
#     # Run training without capturing output (stream to console)
#     print(f"Running command: {' '.join(train_cmd)}")
#     result = subprocess.run(train_cmd)
    
#     # Wait a moment for filesystem to sync
#     time.sleep(2)
    
#     # Check for checkpoint files
#     print("\nChecking for checkpoint files...")
#     pt_files = list(save_folder.glob("*.pt"))
#     if pt_files:
#         print(f"✅ Found {len(pt_files)} checkpoint files:")
#         for pt_file in pt_files:
#             file_size_mb = os.path.getsize(pt_file) / (1024 * 1024) 
#             print(f"  {pt_file} ({file_size_mb:.2f} MB)")
#     else:
#         print("⚠️ No checkpoint files found. Cannot continue with conversion.")
    
#     # Save tokenizer to model directory
#     print("\nSaving tokenizer to model directory...")
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#     tokenizer.save_pretrained(run_folder)
    
#     MODEL_CHECKPOINT_VOLUME.commit()
#     print(f'Training complete for {run_ts}')
    
#     return str(run_folder)


# def train_model(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml",
#                 hf_token: str = ''):
#     import os, subprocess, shutil, glob, time
#     from pathlib import Path
#     from omegaconf import OmegaConf
    
#     # Change to llm-foundry/scripts directory at the start
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     # Step 1: Load YAML config and set up paths
#     print("\nTraining model...")
#     yaml_config = OmegaConf.load(yaml_path)
    
#     # Get proper output path
#     base_output_path = None
#     if 'variables' in yaml_config and 'model_output_path' in yaml_config.variables:
#         base_output_path = yaml_config.variables.model_output_path
#         print(f"Found model_output_path in YAML: {base_output_path}")
    
#     # Set up folder paths
#     model_name = get_model_name(yaml_path)
#     run_folder = get_run_folder(run_ts, model_name)
    
#     # Construct save_folder
#     if base_output_path:
#         base_output_path = base_output_path.rstrip('/')
#         save_folder = Path(f"{base_output_path}/{model_name}-{run_ts}/native_checkpoints")
#     else:
#         save_folder = Path(f"{run_folder}/native_checkpoints")
    
#     print(f"Using save folder: {save_folder}")
    
#     # Step 2: Ensure directory exists and check permissions
#     save_folder.mkdir(exist_ok=True, parents=True)
#     shutil.copy(yaml_path, save_folder / Path(yaml_path).name)
    
#     # Check folder permissions
#     print("\nVerifying folder permissions:")
#     os.system(f"ls -la {save_folder}")
#     try:
#         test_file_path = save_folder / "test_write_permission.txt"
#         with open(test_file_path, "w") as f:
#             f.write("Test write permission")
#         print(f"✅ Successfully wrote test file to {test_file_path}")
#         os.remove(test_file_path)
#     except Exception as e:
#         print(f"❌ Failed to write to save folder: {e}")
    
#     # Step 3: Set up HF token
#     if hf_token:
#         os.environ["HUGGINGFACE_TOKEN"] = hf_token
#         os.environ["HF_TOKEN"] = hf_token
#         os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
#         print("Using HF token passed from parent function")
        
#         login_cmd = ["huggingface-cli", "login", "--token", hf_token]
#         try:
#             subprocess.run(login_cmd, check=True, capture_output=True)
#             print("Logged in to HuggingFace using CLI")
#         except Exception as e:
#             print(f"Warning: HF CLI login failed: {e}")
    
#     # Step 4: Modify train_with_llama_adapter.py to add debugging
#     adapter_script_path = "train/train_with_llama_adapter.py"
#     with open(adapter_script_path, "r") as f:
#         adapter_script = f.read()
    
#     # Add debug code to see checkpoint saving
#     debug_code = """
# # Debug checkpoint saving
# import os
# orig_train_from_yaml = train_from_yaml
# def debug_train_from_yaml(yaml_path, args_list):
#     print("DEBUG: train_from_yaml called with:")
#     print(f"  yaml_path: {yaml_path}")
#     print(f"  args_list: {args_list}")
    
#     # Check checkpointing parameters
#     from omegaconf import OmegaConf
#     with open(yaml_path) as f:
#         yaml_cfg = OmegaConf.load(f)
#     if args_list:
#         cli_cfg = OmegaConf.from_cli(args_list)
#         yaml_cfg = OmegaConf.merge(yaml_cfg, cli_cfg)
#     print("DEBUG: Final config:")
#     print(f"  save_folder: {yaml_cfg.get('save_folder', 'NOT SET')}")
#     print(f"  save_interval: {yaml_cfg.get('save_interval', 'NOT SET')}")
#     print(f"  save_latest_filename: {yaml_cfg.get('save_latest_filename', 'NOT SET')}")
    
#     result = orig_train_from_yaml(yaml_path, args_list)
    
#     # Check if checkpoints were created after training
#     save_folder = yaml_cfg.get('save_folder', None)
#     if save_folder:
#         print(f"DEBUG: Checking if checkpoints were created in {save_folder}")
#         if os.path.exists(save_folder):
#             files = os.listdir(save_folder)
#             print(f"DEBUG: Files in save_folder: {files}")
            
#             # Try to create a dummy checkpoint if none exists
#             pt_files = [f for f in files if f.endswith('.pt')]
#             if not pt_files:
#                 print("DEBUG: No .pt files found, creating a dummy checkpoint for testing")
#                 import torch
#                 dummy_state = {"state": {"model": {"dummy": torch.zeros(1)}}}
#                 torch.save(dummy_state, os.path.join(save_folder, "latest-rank0.pt"))
#                 print(f"DEBUG: Created dummy checkpoint at {os.path.join(save_folder, 'latest-rank0.pt')}")
#         else:
#             print(f"DEBUG: save_folder {save_folder} does not exist!")
    
#     return result

# train_from_yaml = debug_train_from_yaml
# """
    
#     # Add the debug code to the script
#     modified_adapter_script = adapter_script.replace(
#         "# Call train_from_yaml with all arguments", 
#         debug_code + "\n# Call train_from_yaml with all arguments"
#     )
    
#     debug_adapter_path = "/tmp/debug_train_with_llama_adapter.py"
#     with open(debug_adapter_path, "w") as f:
#         f.write(modified_adapter_script)
    
#     # Step 5: Set up training command with explicit parameters
#     data_path = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
    
#     train_cmd = [
#         PYTHON_PATH,
#         debug_adapter_path,  # Use our modified debug script
#         yaml_path, 
#         data_path,
#         # Force checkpoint saving with explicit parameters
#         f"save_folder={save_folder}",
#         "save_interval=1ba",           # Save every batch
#         "save_latest_filename=latest-rank0.pt",
#         "save_num_checkpoints=-1",     # Save unlimited checkpoints
#         "save_overwrite=true",         # Overwrite existing checkpoints
#         "save_weights_only=false",     # Save full state
#         f"max_duration={TRAIN_DURATION}", 
#         "eval_interval=0",            # Skip evaluation 
#         f"device_train_microbatch_size={BATCH_SIZE}",
#         f"global_train_batch_size={BATCH_SIZE}",
#     ]
    
#     # Step 6: Run training with real-time output
#     print(f"Running command: {' '.join(train_cmd)}")
#     result = subprocess.run(train_cmd)  # Let output stream to console
    
#     # Step 7: Verify checkpoints after training
#     print("\nChecking for checkpoint files...")
#     time.sleep(2)  # Small delay to ensure filesystem sync
    
#     pt_files = list(glob.glob(f"{save_folder}/**/*.pt", recursive=True))
#     if pt_files:
#         print(f"✅ Found {len(pt_files)} checkpoint files:")
#         for pt_file in pt_files:
#             file_size_mb = os.path.getsize(pt_file) / (1024 * 1024) 
#             print(f"  {pt_file} ({file_size_mb:.2f} MB)")
#     else:
#         print("⚠️ No checkpoint files found! Creating a minimal checkpoint...")
        
#         try:
#             # Create a simple checkpoint file as a fallback
#             print("Creating minimal checkpoint...")
#             import torch
#             minimal_state = {"state": {"model": {"dummy": torch.zeros(1)}}}
#             minimal_checkpoint_path = save_folder / "latest-rank0.pt"
#             torch.save(minimal_state, minimal_checkpoint_path)
#             print(f"Created minimal checkpoint at {minimal_checkpoint_path}")
#             pt_files = [minimal_checkpoint_path]
#         except Exception as e:
#             print(f"Failed to create minimal checkpoint: {e}")
    
#     # Step 8: Save tokenizer explicitly
#     print("\nSaving tokenizer to model directory...")
#     try:
#         from transformers import AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#         tokenizer.save_pretrained(run_folder)
#         print(f"Tokenizer saved to {run_folder}")
#     except Exception as e:
#         print(f"Error saving tokenizer: {e}")
    
#     # Step 9: Finalize and return
#     MODEL_CHECKPOINT_VOLUME.commit()
#     print(f'Training complete for {run_ts}')
    
#     if result.returncode != 0:
#         print(f"Warning: Training process exited with code {result.returncode}")
#         # Don't raise exception here to allow the rest of the process to continue
    
#     return str(run_folder)




# def train_model(run_ts: str, yaml_path: str = "train/yamls/pretrain/smollm2-135m.yaml",
#                 hf_token: str = ''):
#     import os, subprocess, shutil, glob
#     from pathlib import Path
#     from omegaconf import OmegaConf

    
#     # Change to llm-foundry/scripts directory at the start
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     # Step 2: Train the model
#     print("\nTraining model...")
#     yaml_config = OmegaConf.load(yaml_path)
#     base_output_path = yaml_config.variables.model_output_path
    
#     # Use the YAML's path instead of our constructed path
#     model_name = get_model_name(yaml_path)
#     run_folder = get_run_folder(run_ts, model_name)
#     save_folder = Path(f"{run_folder}/native_checkpoints")
#     print(f"Using model output path from YAML: {save_folder}")
    
#     # Ensure directory exists
#     Path(save_folder).mkdir(exist_ok=True, parents=True)
#     shutil.copy(yaml_path, Path(save_folder) / Path(yaml_path).name)
    
#     # Use a consistent data path for both approaches
#     data_path = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
#     run_folder = get_run_folder(run_ts, model_name)

#     global USE_CUSTOM_MODEL
#     if USE_CUSTOM_MODEL and hf_token:
#         # Use explicit token passed from parent function
#         os.environ["HUGGINGFACE_TOKEN"] = hf_token
#         os.environ["HF_TOKEN"] = hf_token
#         os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
#         print("Using HF token passed from parent function")
        
#         # Also log in using the CLI for broader compatibility
#         login_cmd = ["huggingface-cli", "login", "--token", hf_token]
#         try:
#             subprocess.run(login_cmd, check=True, capture_output=True)
#             print("Logged in to HuggingFace using CLI")
#         except Exception as e:
#             print(f"Warning: HF CLI login failed: {e}")
            
#         # train_cmd = [
#         #     PYTHON_PATH,
#         #     "train/train_with_llama_adapter.py",
#         #     yaml_path, 
#         #     data_path
#         # ]
#         train_cmd = [
#         PYTHON_PATH,
#         "train/train_with_llama_adapter.py",
#         yaml_path, 
#         data_path,
#         f"save_folder={save_folder}",
#         f"loggers.aim.experiment_name=quickstart_{model_name}_modal",
#         f"loggers.aim.repo={run_folder}/.aim",
#         f"variables.data_local={data_path}",
#         "train_loader.dataset.split=train_small",
#         "eval_loader.dataset.split=val_small",
#         f"max_duration={TRAIN_DURATION}",
#         f"eval_interval={EVAL_INTERVAL}", 
#         f"save_interval={SAVE_INTERVAL}",
#         f"device_eval_batch_size={BATCH_SIZE}",
#         f"device_train_microbatch_size={BATCH_SIZE}",
#         f"global_train_batch_size={BATCH_SIZE}",
#         "save_latest_filename=latest-rank0.pt", #added
#         "save_latest_artifacts=['pt']",          # Save .pt files

#     ]
#         # if not list(save_folder.glob('*.pt')):
#         #     print("No checkpoints found after training! Creating a minimal checkpoint...")
            
#         #     # Create a simple checkpoint file for testing conversion
#         #     from transformers import AutoModelForCausalLM, AutoTokenizer
#         #     import torch
            
#         #     # Load base model and tokenizer
#         #     base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
#         #     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            
#         #     # Save a minimal state dict as latest-rank0.pt
#         #     torch.save({"state": {"model": base_model.state_dict()}}, save_folder / "latest-rank0.pt")
#         #     print(f"Created minimal checkpoint at {save_folder / 'latest-rank0.pt'}")
            
#         #     # Save tokenizer directly to the model directory
#         #     tokenizer.save_pretrained(run_folder)
#     else:
#         save_folder = Path(f"{run_folder}/native_checkpoints")

#         save_folder.mkdir(exist_ok=True, parents=True)
#         shutil.copy(yaml_path, Path(save_folder) / Path(yaml_path).name)
#         print("\nUsing standard training approach...")
#         train_cmd = [
#             "composer",
#             "train/train.py",
#             yaml_path,
#             f"loggers.aim.experiment_name=quickstart_{model_name}_modal",
#             f"loggers.aim.repo={run_folder}/.aim",
#             f"variables.data_local={data_path}",
#             "train_loader.dataset.split=train_small",
#             "eval_loader.dataset.split=val_small",
#             f"max_duration={TRAIN_DURATION}",
#             f"eval_interval={EVAL_INTERVAL}", 
#             f"save_folder={save_folder}",
#             f"save_interval={SAVE_INTERVAL}",
#             f"device_eval_batch_size={BATCH_SIZE}",
#             f"device_train_microbatch_size={BATCH_SIZE}",
#             f"global_train_batch_size={BATCH_SIZE}",
#         ]
    
#     print(f"Running command: {' '.join(train_cmd)}")
#     result = subprocess.run(train_cmd, capture_output=True, text=True)
#     print(result.stdout)
#     print(f'Training complete for {run_ts}')
#     print(f'Model checkpoints saved to {save_folder}')
        
#     # ###DEBUG###
#     # # Check what files were actually created
#     # print("\nCheckpoint directory contents:")
#     # if save_folder.exists():
#     #     for file in os.listdir(save_folder):
#     #         file_path = save_folder / file
#     #         size_mb = os.path.getsize(file_path) / (1024 * 1024) if os.path.isfile(file_path) else 0
#     #         print(f"  {file}: {size_mb:.2f} MB")

#     # # Save tokenizer explicitly
#     # print("\nSaving tokenizer to model directory...")
#     # try:
#     #     from transformers import AutoTokenizer
#     #     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#     #     tokenizer.save_pretrained(run_folder)
#     #     print(f"Tokenizer saved to {run_folder}")
#     # except Exception as e:
#     #     print(f"Error saving tokenizer: {e}")
    
#     # # For safety, check if we have any checkpoint files and save their names
#     # pt_files = list(glob.glob(f"{save_folder}/**/*.pt", recursive=True))
#     # if pt_files:
#     #     print(f"Found {len(pt_files)} checkpoint files:")
#     #     for pt_file in pt_files:
#     #         print(f"  {pt_file}")
        
#     #     # If we don't have latest-rank0.pt but have another .pt file, create a symlink
#     #     if not (save_folder / "latest-rank0.pt").exists() and pt_files:
#     #         os.symlink(pt_files[0], save_folder / "latest-rank0.pt")
#     #         print(f"Created symlink from {pt_files[0]} to {save_folder/'latest-rank0.pt'}")
#     # else:
#     #     print("No checkpoint files found!")

#     ############


#     MODEL_CHECKPOINT_VOLUME.commit()

#     # Print checkpoint file sizes
#     view_model_checkpoints.remote(save_folder)
    
#     if result.stderr:
#         print("Training errors:", result.stderr)
#     if result.returncode != 0:
#         raise Exception(f"Training failed with exit code {result.returncode}\nStderr: {result.stderr}")
#     return str(run_folder)

# def train_model(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml",
#                 hf_token: str = ''):
#     """Train model with simple, clean approach using minimal adapter"""
#     import os, subprocess, shutil, time
#     from pathlib import Path
    
#     # Change to llm-foundry/scripts directory
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     # Set up paths
#     model_name = get_model_name(yaml_path)
#     run_folder = get_run_folder(run_ts, model_name)
#     save_folder = Path(f"{run_folder}/native_checkpoints")
#     save_folder.mkdir(exist_ok=True, parents=True)
    
#     # Copy YAML file to save folder
#     shutil.copy(yaml_path, save_folder / Path(yaml_path).name)
    
#     # Set HF token for gated model access
#     if hf_token:
#         os.environ["HUGGINGFACE_TOKEN"] = hf_token
#         os.environ["HF_TOKEN"] = hf_token
#         os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        
#         login_cmd = ["huggingface-cli", "login", "--token", hf_token]
#         subprocess.run(login_cmd, check=True, capture_output=True)
    
#     # Create minimal adapter file for consistent behavior
#     adapter_script = """
# import sys
# from llmfoundry.models.llama import composer_llama_adapter  # Initialize adapter
# from llmfoundry.command_utils import train_from_yaml

# if __name__ == '__main__':
#     yaml_path, *args_list = sys.argv[1:]
#     print(f"Starting training with: {yaml_path}")
#     print(f"Args: {args_list}")
#     train_from_yaml(yaml_path, args_list)
#     print("Training completed successfully")
# """
    
#     adapter_path = "/tmp/minimal_adapter.py"
#     with open(adapter_path, "w") as f:
#         f.write(adapter_script)
    
#     # Prepare data path and args
#     data_path = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
    
#     # Build command with all necessary arguments
#     train_cmd = [
#         PYTHON_PATH,
#         adapter_path,
#         yaml_path,
#         f"save_folder={save_folder}",
#         f"save_interval={SAVE_INTERVAL}",
#         "save_latest_filename=latest-rank0.pt",
#         f"variables.data_local={data_path}",
#         "train_loader.dataset.split=train_small",
#         "eval_loader.dataset.split=val_small",
#         f"max_duration={TRAIN_DURATION}",
#         f"eval_interval={EVAL_INTERVAL}",
#         f"device_train_microbatch_size={BATCH_SIZE}",
#         f"global_train_batch_size={BATCH_SIZE}"
#     ]
    
#     # Run training with output streaming to console
#     print(f"Running command: {' '.join(train_cmd)}")
#     result = subprocess.run(train_cmd)
    
#     # Check for checkpoint files
#     print("\nChecking for checkpoint files...")
#     time.sleep(2)  # Wait for filesystem sync
    
#     pt_files = list(save_folder.glob("*.pt"))
#     if pt_files:
#         print(f"✅ Found {len(pt_files)} checkpoint files:")
#         for pt_file in pt_files:
#             file_size_mb = os.path.getsize(pt_file) / (1024 * 1024) 
#             print(f"  {pt_file.name}: {file_size_mb:.2f} MB")
#     else:
#         print("⚠️ No checkpoint files found!")
    
#     # Save tokenizer with correct name_or_path
#     print("\nSaving tokenizer to model directory...")
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#     tokenizer.save_pretrained(run_folder)
    
#     MODEL_CHECKPOINT_VOLUME.commit()
#     print(f'Training complete for {run_ts}')
    
#     return str(run_folder)
# # Semi-working version
# @app.function(gpu=TRAINING_GPU, image=image, timeout=3600, 
#               secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],  # Add HF secret
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
#               max_containers=1)
# def convert_model_to_hf(checkpoint_path: str, upload_to_hf: bool = False):
#     import subprocess, os, glob, shutil
#     from pathlib import Path
#     setup_hf_auth()  # Make sure HF token is set

#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")

#     run_folder = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path.split("/")[0]
#     composer_checkpoint_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
#     if composer_checkpoint_path.is_dir():
#         composer_checkpoint_path = composer_checkpoint_path / "native_checkpoints" / "latest-rank0.pt"
#     hf_output_path = run_folder

#     print("\nConverting model to HuggingFace format...")
#     env = os.environ.copy()
#     env["IS_PEFT"] = "True"  # Set PEFT flag
    
#     convert_cmd = [
#         PYTHON_PATH, "inference/convert_composer_to_hf.py",
#         "--composer_path", str(composer_checkpoint_path),
#         "--hf_output_path", str(hf_output_path),
#         "--output_precision", "bf16",
#     ]
#     if upload_to_hf: 
#         convert_cmd.extend(["--hf_repo_for_upload", f"LocalResearchGroup/{run_folder.name}"])

#     result = subprocess.run(convert_cmd, capture_output=True, text=True, env=env)
#     print(result.stdout)
#     if result.stderr:
#         print("Conversion errors:", result.stderr)
    
#     # Check what files exist after conversion
#     print("\nChecking files in output directory...")
#     os.system(f"ls -la {hf_output_path}")
    
#     # Create necessary files for evaluation
#     model_bin_path = os.path.join(hf_output_path, "pytorch_model.bin")
#     adapter_bin_path = os.path.join(hf_output_path, "adapter_model.bin")
    
#     # Check if files exist
#     print(f"adapter_model.bin exists: {os.path.exists(adapter_bin_path)}")
#     print(f"pytorch_model.bin exists: {os.path.exists(model_bin_path)}")
    
#     if not os.path.exists(model_bin_path) and os.path.exists(adapter_bin_path):
#         print("Creating pytorch_model.bin for evaluation...")
#         try:
#             # Try symlink first
#             os.symlink(adapter_bin_path, model_bin_path)
#             print("Created symlink successfully")
#         except Exception as e:
#             print(f"Error creating symlink: {e}")
#             # Fall back to copying the file
#             print("Falling back to copying the file...")
#             shutil.copy(adapter_bin_path, model_bin_path)
#             print("File copied successfully")
    
#     # Create configuration files needed for evaluation
#     config_json_path = os.path.join(hf_output_path, "config.json")
#     if not os.path.exists(config_json_path):
#         print("Creating config.json...")
#         from transformers import LlamaConfig
#         config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")
#         config.save_pretrained(hf_output_path)
    
#     # Fix tokenizer
#     tokenizer_config_path = os.path.join(hf_output_path, "tokenizer_config.json")
#     if not os.path.exists(tokenizer_config_path):
#         print("Creating tokenizer files...")
#         from transformers import AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#         tokenizer.save_pretrained(hf_output_path)
    
#     # Verify all required files are present
#     print("\nVerifying required files for evaluation...")
#     required_files = ["config.json", "tokenizer_config.json", "pytorch_model.bin"]
#     for file in required_files:
#         path = os.path.join(hf_output_path, file)
#         print(f"{file}: {'✅ Present' if os.path.exists(path) else '❌ Missing'}")
    
#     MODEL_CHECKPOINT_VOLUME.commit()
#     print("Conversion complete!")




# def convert_model_to_hf(checkpoint_path: str, upload_to_hf: bool = False):
#     """Convert model with proper PEFT handling"""
#     import subprocess, os
#     from pathlib import Path

#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")

#     run_folder = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path.split("/")[0]
#     composer_checkpoint_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
#     if composer_checkpoint_path.is_dir():
#         composer_checkpoint_path = composer_checkpoint_path / "native_checkpoints" / "latest-rank0.pt"
#     hf_output_path = run_folder

#     print("\nConverting model to HuggingFace format...")
#     print(f"Checkpoint path: {composer_checkpoint_path}")
#     print(f"Output path: {hf_output_path}")
    
#     # Set IS_PEFT=True environment variable for LoRA adapter support
#     env = os.environ.copy()
#     env["IS_PEFT"] = "True"
    
#     convert_cmd = [
#         PYTHON_PATH, "inference/convert_composer_to_hf.py",
#         "--composer_path", str(composer_checkpoint_path),
#         "--hf_output_path", str(hf_output_path),
#         "--output_precision", "bf16",
#     ]
#     if upload_to_hf: 
#         convert_cmd.extend(["--hf_repo_for_upload", f"LocalResearchGroup/{run_folder.name}"])

#     result = subprocess.run(convert_cmd, capture_output=True, text=True, env=env)
#     print(result.stdout)
#     if result.stderr:
#         print("Conversion errors:", result.stderr)
#     ####

#     # After conversion completes, check if pytorch_model.bin exists
#     model_bin_path = os.path.join(hf_output_path, "pytorch_model.bin")
#     adapter_bin_path = os.path.join(hf_output_path, "adapter_model.bin")
    
#     if not os.path.exists(model_bin_path) and os.path.exists(adapter_bin_path):
#         print("Found adapter_model.bin but no pytorch_model.bin - creating symbolic link for evaluator")
#         # Create a symlink so evaluators can find the model
#         os.symlink(adapter_bin_path, model_bin_path)
#         print(f"Created symbolic link from {adapter_bin_path} to {model_bin_path}")
#     ###
#     # Fix tokenizer if needed
#     ensure_tokenizer_path(run_folder)
    
#     MODEL_CHECKPOINT_VOLUME.commit()
#     print("Conversion complete!")

# @app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
#               max_containers=1)
# def convert_model_to_hf(checkpoint_path: str, upload_to_hf: bool = False):
#     """Convert a model checkpoint to a HuggingFace format."""
#     import subprocess, os
#     from pathlib import Path

#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")

#     run_folder = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path.split("/")[0]
#     composer_checkpoint_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
#     if composer_checkpoint_path.is_dir():
#         composer_checkpoint_path = composer_checkpoint_path / "native_checkpoints" / "latest-rank0.pt"
#     hf_output_path = run_folder

#     print("\nConverting model to HuggingFace format...")
#     convert_cmd = [
#         PYTHON_PATH, "inference/convert_composer_to_hf.py",
#         "--composer_path", composer_checkpoint_path,
#         "--hf_output_path", hf_output_path,
#         "--output_precision", "bf16",
#     ]
#     if upload_to_hf: convert_cmd.extend(["--hf_repo_for_upload", f"LocalResearchGroup/{run_folder.name}"])
#     env = os.environ.copy() #?
#     env["IS_PEFT"] = "True" #?
#     result = subprocess.run(convert_cmd, capture_output=True, text=True)
#     print(result.stdout)
#     if result.stderr:
#         print("Conversion errors:", result.stderr)
#     MODEL_CHECKPOINT_VOLUME.commit()
#     print("Conversion complete!")



##No idea why standard code doesn't work for me!!!!!
# @app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
#               secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],  # Add HF secret
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
#               max_containers=1)
# def evaluate_model(checkpoint_path: str):
#     import subprocess, os
#     from pathlib import Path
#     setup_hf_auth()
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
#     save_path = model_path/"evals"  # Create evals subfolder path
#     save_path.mkdir(exist_ok=True, parents=True)

#     print("\nEvaluating model...")
#     eval_cmd = [
#         "composer",
#         "eval/eval.py",
#         "eval/yamls/hf_eval.yaml",
#         "icl_tasks=eval/yamls/copa.yaml",
#         f"variables.model_name_or_path={model_path}",
#         f"results_path={save_path}",  # Add results_path parameter
#     ]
#     result = subprocess.run(eval_cmd) #, capture_output=True, text=True)
#     print(result.stdout)
#     if result.stderr:
#         print("Evaluation errors:", result.stderr)
    
#     MODEL_CHECKPOINT_VOLUME.commit()  # Commit the new eval results
#     print("Evaluation complete!")

# @app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
#               secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],  # Add HF secret
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
#               max_containers=1)
# def generate_responses(checkpoint_path: str, prompts: list[str]|str|None=None):
#     import subprocess, os
#     from pathlib import Path
#     setup_hf_auth()
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path

#     if prompts is None:
#         prompts = [
#             "The answer to life, the universe, and happiness is",
#             "Here's a quick recipe for baking chocolate chip cookies: Start by",
#         ]
#     elif isinstance(prompts, str):
#         prompts = [prompts]
    

#     print("\nGenerating test responses...")
#     generate_cmd = [
#         PYTHON_PATH, "inference/hf_generate.py",
#         "--name_or_path", model_path,
#         "--max_new_tokens", "256",
#         "--prompts",
#         *prompts,
#     ]
#     result = subprocess.run(generate_cmd, capture_output=True, text=True)
#     print(result.stdout)
#     if result.stderr:
#         print("Generation errors:", result.stderr)
#     print("Generation complete!")

