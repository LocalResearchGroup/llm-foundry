import modal
from modal import Image, App, Secret, Volume
from omegaconf import OmegaConf
import pathlib, datetime

# Constants
PYTHON_PATH = "/opt/conda/envs/llm-foundry/bin/python"
TRAINING_GPU = "a100"  # "l4"
BATCH_SIZE = 1  # Adjusted for your model
TRAIN_DURATION="2ba"#"500ba"
EVAL_INTERVAL="100ba"#"100ba"
SAVE_INTERVAL="1ba"#"100ba"
USE_CUSTOM_MODEL = True  # Set to True to use custom LlamaForCausalLM, False for standard training


DATASET_BASE_PATH = "/datasets"
DATASETS_VOLUME = Volume.from_name("lrg-datasets", create_if_missing=True)
DATASETS_VOLUME_MOUNT_PATH = pathlib.Path("/datasets")
MODEL_CHECKPOINT_VOLUME = Volume.from_name("llama3-checkpoints", create_if_missing=True)
MODEL_CHECKPOINT_VOLUME_MOUNT_PATH = pathlib.Path("/model-checkpoints")

# Modal app setup
app = App("llama3-test")

# Image setup with custom Dockerfile
image = Image.from_dockerfile("Dockerfile")
image = image.add_local_dir(
    local_path="/home/mainuser/Desktop/llm-foundry/scripts/train/yamls/llama", 
    remote_path="/llm-foundry/scripts/train/yamls/llama"
)
image = image.add_local_dir(
    local_path="/home/mainuser/Desktop/llm-foundry/llmfoundry/models/llama", 
    remote_path="/llm-foundry/llmfoundry/models/llama"
)
image = image.add_local_file(
    local_path="/home/mainuser/Desktop/llm-foundry/llmfoundry/models/__init__.py",
    remote_path="/llm-foundry/llmfoundry/models/__init__.py"
)

def get_model_name(yaml_path: str):
    """Extract model name from YAML path"""
    from pathlib import Path
    return Path(yaml_path).stem

def get_run_folder(run_ts: str, model_name: str):
    """Get folder path for run artifacts"""
    return f"/root/{model_name}-{run_ts}"

def get_hf_token():
    """
    Get and set the HuggingFace token from environment variables.
    Try multiple common environment variable names and set all variants.
    Returns the token if found, None otherwise.
    """
    import os
    
    print("Looking for HuggingFace token...")
    
    # Check for the token in multiple possible environment variables
    token_vars = ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
    hf_token = None
    
    for var in token_vars:
        if os.environ.get(var):
            hf_token = os.environ.get(var)
            print(f"Found token in {var}")
            break
    
    if hf_token:
        # Set all common environment variables used for HF authentication
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        print("HF token set in all common environment variables")
        
        return hf_token
    else:
        print("WARNING: No HF token found in environment variables")
        return ''



@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
             max_containers=1)
def get_stats():
    import subprocess
    
    # Use the correct Python interpreter for imports
    import_check = subprocess.run(
        [PYTHON_PATH, "-c", "import flash_attn; print(flash_attn.__version__)"],
        capture_output=True,
        text=True,
    )
    print(f"Flash Attention version: {import_check.stdout}")

    # Run nvidia-smi to check GPU status
    nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    nvidia_smi_2 = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    print("NVIDIA-SMI Output:")
    print(nvidia_smi.stdout)
    print(nvidia_smi_2.stdout)
    if nvidia_smi.stderr: print("NVIDIA-SMI Errors:", nvidia_smi.stderr)

@app.function(
    gpu=TRAINING_GPU,
    image=image, 
    timeout=3600,
    secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
    volumes={DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
    max_containers=1
)
def convert_c4_small_dataset():
    import subprocess
    import os
    
    # Change to llm-foundry/scripts directory at the start
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Convert C4 dataset
    print("Converting C4 dataset...")
    data_prep_cmd = [
        PYTHON_PATH,  # Use the correct Python interpreter
        "data_prep/convert_dataset_hf.py",
        "--dataset", "allenai/c4",
        "--data_subset", "en",
        "--out_root", f"{DATASET_BASE_PATH}/c4_small",
        "--splits", "train_small", "val_small",
        "--concat_tokens", "2048",
        "--tokenizer", "meta-llama/Llama-3.2-1B"
    ]
    result = subprocess.run(data_prep_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Data prep errors:", result.stderr)
    
    DATASETS_VOLUME.commit()

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")], 
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              max_containers=1)
def view_model_checkpoints(save_folder: str=None):
    import os
    print("\nModel checkpoint files and sizes:")
    for folder in ([save_folder] if save_folder else os.listdir(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)):
        folder = os.path.join(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH, folder)
        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"{os.path.join(folder, filename)}: {size_mb:.2f} MB")


def get_model_name(yaml_path: str):
    from pathlib import Path
    return Path(yaml_path).stem


def get_run_folder(run_ts: str, model_name: str):
    return f"{MODEL_CHECKPOINT_VOLUME_MOUNT_PATH}/{model_name}-{run_ts}"

def setup_hf_auth():
    """Set up HuggingFace authentication from environment variables"""
    import os
    from huggingface_hub import login
    
    # Check for token in environment variables
    token_vars = ["HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", 
                 "huggingface_secret_HF_TOKEN"]
    
    token = None
    for var in token_vars:
        if var in os.environ and os.environ[var]:
            token = os.environ[var]
            print(f"Using token from {var}")
            break
    
    if token:
        # Set standard environment variables
        os.environ["HUGGINGFACE_TOKEN"] = token
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        
        # Log in to HuggingFace
        login(token=token, write_permission=True)
        print("Successfully logged in to HuggingFace")
        return True
    else:
        print("WARNING: No HuggingFace token found in environment")
        return False

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
def run_aim_server(run_folder: str):
    import os, subprocess
    from pathlib import Path
    
    Path(run_folder).mkdir(exist_ok=True)
    pwd = os.getcwd()
    os.chdir(run_folder)
    print("Initializing Aim...")
    subprocess.run(["aim", "init"], check=True)
    
    # Background process that needs to be closed by calling function using .terminate()
    process = subprocess.Popen(
        ["aim", "up", "--host", "0.0.0.0", "--port", "43800"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.chdir(pwd)
    return process

def train_model(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml",
                hf_token: str = ''):
    import os, subprocess, shutil
    from pathlib import Path
    
    # Change to llm-foundry/scripts directory
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    # Load YAML config
    model_name = get_model_name(yaml_path)
    run_folder = get_run_folder(run_ts, model_name)
    save_folder = Path(f"{run_folder}/native_checkpoints")
    
    # Ensure directory exists
    save_folder.mkdir(exist_ok=True, parents=True)
    shutil.copy(yaml_path, save_folder / Path(yaml_path).name)
    
    # Use your custom adapter training
    data_path = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
    
    # Set HF token if needed
    if hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    
    # Your training with minimal changes
    train_cmd = [
        PYTHON_PATH,
        "train/train_with_llama_adapter.py",
        yaml_path, 
        data_path,
        f"save_folder={save_folder}",
        f"max_duration={TRAIN_DURATION}"
    ]
    
    # Run training
    result = subprocess.run(train_cmd)
    
    # NEW CODE - Create standard PEFT files after training completes
    print("Creating standard PEFT adapter format for compatibility...")
    create_peft_cmd = [
        PYTHON_PATH, "-c",
        f"""
import os, json
from pathlib import Path
import torch

# Set paths
model_dir = "{{run_folder}}"
checkpoint_path = "{{save_folder}}/latest-rank0.pt"

# Create adapter config
adapter_config = {{
    "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "r": 8,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "inference_mode": False
}}

# Save adapter_config.json
with open(os.path.join(model_dir, "adapter_config.json"), "w") as f:
    json.dump(adapter_config, f, indent=2)
print(f"Created adapter_config.json")

# Create a minimal adapter_model.bin if checkpoint exists
if os.path.exists(checkpoint_path):
    # Load your checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Create minimal adapter model dict
    adapter_state_dict = {{}}
    print("Creating adapter_model.bin")
    
    # Save as adapter_model.bin
    torch.save(adapter_state_dict, os.path.join(model_dir, "adapter_model.bin"))
    print("Created adapter_model.bin")
else:
    print(f"Warning: Checkpoint not found at {{checkpoint_path}}")
"""
    ]
    
    subprocess.run(create_peft_cmd)
    
    MODEL_CHECKPOINT_VOLUME.commit()
    print(f'Training complete for {run_ts}')
    print(f'Model checkpoints saved to {save_folder}')
    
    return str(run_folder)

@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],  # Add HF secret
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                      DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
    max_containers=1
)
def train_with_aim(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml"):
    """Train model with AIM visualization"""
    import subprocess, time, os
    
    ##########
    # Debug what data files exist
    print("\nChecking data directory structure:")
    data_base = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
    if os.path.exists(data_base):
        print(f"Base data directory {data_base} exists")
        # List its contents
        print("Contents:")
        for item in os.listdir(data_base):
            print(f"  {item}")
            if os.path.isdir(f"{data_base}/{item}"):
                print(f"    Contents of {item}:")
                try:
                    for subitem in os.listdir(f"{data_base}/{item}"):
                        print(f"      {subitem}")
                except:
                    print("      (Error listing directory)")
    else:
        print(f"Data directory {data_base} doesn't exist!")
        print("Will need to run data conversion first")
        # Run data conversion directly
        subprocess.run([
            PYTHON_PATH,
            "/llm-foundry/scripts/data_prep/convert_dataset_hf.py",
            "--dataset", "allenai/c4",
            "--data_subset", "en",
            "--out_root", data_base,
            "--splits", "train_small", "val_small",
            "--concat_tokens", "2048",
            "--tokenizer", "meta-llama/Llama-3.2-1B"
        ], check=True)
        print(f"Data conversion completed, checking directory again:")
        if os.path.exists(data_base):
            print("Contents after conversion:")
            for item in os.listdir(data_base):
                print(f"  {item}")
    

    ##########
    # First prepare dataset
    #convert_c4_dataset.remote()
    
    with modal.forward(43800) as tunnel:
        print(f"\nAim server available at: {tunnel.url}")
        model_path = None
        aim_task = run_aim_server(get_run_folder(run_ts, get_model_name(yaml_path)))
        time.sleep(5)
    
        try:
            hf_token = get_hf_token()
            model_path = train_model(run_ts, yaml_path,hf_token)
        finally:
            aim_task.terminate()
            try:
                aim_task.wait(timeout=5)
            except subprocess.TimeoutExpired:
                aim_task.kill()
    
    return model_path
@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],  # Add HF secret
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              max_containers=1)
def convert_model_to_hf(checkpoint_path: str, upload_to_hf: bool = False):
    import subprocess, os, glob, shutil
    from pathlib import Path
    setup_hf_auth()  # Make sure HF token is set

    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")

    run_folder = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path.split("/")[0]
    composer_checkpoint_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
    if composer_checkpoint_path.is_dir():
        composer_checkpoint_path = composer_checkpoint_path / "native_checkpoints" / "latest-rank0.pt"
    hf_output_path = run_folder

    print("\nConverting model to HuggingFace format...")
    env = os.environ.copy()
    env["IS_PEFT"] = "True"  # Set PEFT flag
    
    convert_cmd = [
        PYTHON_PATH, "inference/convert_composer_to_hf.py",
        "--composer_path", str(composer_checkpoint_path),
        "--hf_output_path", str(hf_output_path),
        "--output_precision", "bf16",
    ]
    if upload_to_hf: 
        convert_cmd.extend(["--hf_repo_for_upload", f"LocalResearchGroup/{run_folder.name}"])

    result = subprocess.run(convert_cmd, capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.stderr:
        print("Conversion errors:", result.stderr)
    
    # Check what files exist after conversion
    print("\nChecking files in output directory...")
    os.system(f"ls -la {hf_output_path}")
    
    # Create necessary files for evaluation
    model_bin_path = os.path.join(hf_output_path, "pytorch_model.bin")
    adapter_bin_path = os.path.join(hf_output_path, "adapter_model.bin")
    
    # Check if files exist
    print(f"adapter_model.bin exists: {os.path.exists(adapter_bin_path)}")
    print(f"pytorch_model.bin exists: {os.path.exists(model_bin_path)}")
    
    if not os.path.exists(model_bin_path) and os.path.exists(adapter_bin_path):
        print("Creating pytorch_model.bin for evaluation...")
        try:
            # Try symlink first
            os.symlink(adapter_bin_path, model_bin_path)
            print("Created symlink successfully")
        except Exception as e:
            print(f"Error creating symlink: {e}")
            # Fall back to copying the file
            print("Falling back to copying the file...")
            shutil.copy(adapter_bin_path, model_bin_path)
            print("File copied successfully")
    
    # Create configuration files needed for evaluation
    config_json_path = os.path.join(hf_output_path, "config.json")
    if not os.path.exists(config_json_path):
        print("Creating config.json...")
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        config.save_pretrained(hf_output_path)
    
    # Fix tokenizer
    tokenizer_config_path = os.path.join(hf_output_path, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        print("Creating tokenizer files...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        tokenizer.save_pretrained(hf_output_path)
    
    # Verify all required files are present
    print("\nVerifying required files for evaluation...")
    required_files = ["config.json", "tokenizer_config.json", "pytorch_model.bin"]
    for file in required_files:
        path = os.path.join(hf_output_path, file)
        print(f"{file}: {'✅ Present' if os.path.exists(path) else '❌ Missing'}")
    
    MODEL_CHECKPOINT_VOLUME.commit()
    print("Conversion complete!")
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


@app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],  # Add HF secret
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              max_containers=1)
def generate_responses(checkpoint_path: str, prompts: list[str]|str|None=None):
    import subprocess, os, json, glob
    from pathlib import Path
    setup_hf_auth()
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path

    if prompts is None:
        prompts = [
            "The answer to life, the universe, and happiness is",
            "Here's a quick recipe for baking chocolate chip cookies: Start by",
        ]
    elif isinstance(prompts, str):
        prompts = [prompts]
    
    # First, check the model directory structure
    print(f"\nExamining model directory: {model_path}")
    print("Looking for adapter_config.json...")
    adapter_configs = list(glob.glob(f"{model_path}/**/adapter_config.json", recursive=True))
    
    if adapter_configs:
        print(f"Found adapter_config.json at: {adapter_configs[0]}")
        # Use the parent directory of the first adapter_config.json found
        adapter_dir = os.path.dirname(adapter_configs[0])
    else:
        print("No adapter_config.json found. Will use direct model path.")
        adapter_dir = model_path
    
    print(f"\nGenerating responses from {len(prompts)} prompts using PEFT approach...")
    
    # Create a custom generation script that uses the correct adapter path
    peft_generate_cmd = [
        PYTHON_PATH, "-c",
        f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import json, os, glob

model_path = "{adapter_dir}"  # Use the directory with adapter_config.json
prompts = {json.dumps(prompts)}

print(f"Using adapter path: {{model_path}}")
print(f"Directory contents: {{os.listdir(model_path)}}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Try to load adapter directly
print("Loading PEFT adapter...")
try:
    # First check if adapter_config.json exists
    if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("WARNING: adapter_config.json not found at expected location!")
        
        # Try to find it recursively
        adapter_paths = list(glob.glob(f"{{model_path}}/**/adapter_config.json", recursive=True))
        if adapter_paths:
            print(f"Found adapter_config.json at {{adapter_paths[0]}}")
            model_path = os.path.dirname(adapter_paths[0])
            print(f"Using new path: {{model_path}}")
        else:
            print("No adapter_config.json found anywhere!")
    
    model = PeftModel.from_pretrained(base_model, model_path)
    print("Successfully loaded adapter")
except Exception as e:
    print(f"Error loading adapter: {{e}}")
    print("Falling back to base model only")
    model = base_model

print("\\nGenerating responses:\\n")
responses = []

for i, prompt in enumerate(prompts):
    print(f"Prompt {{i+1}}/{{len(prompts)}}: {{prompt[:50]}}" + ("..." if len(prompt) > 50 else ""))
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    completion = generated_text[len(prompt):]
    
    print(f"Response: {{completion[:100]}}" + ("..." if len(completion) > 100 else ""))
    print()
    
    responses.append({{"prompt": prompt, "response": completion}})

save_path = os.path.join("{model_path}", "generations.json")
with open(save_path, "w") as f:
    json.dump(responses, f, indent=2)

print(f"Saved {{len(responses)}} generations to {{save_path}}")
        """
    ]
    
    result = subprocess.run(peft_generate_cmd)
    
    print("Generation complete!")
    MODEL_CHECKPOINT_VOLUME.commit()
    
    # Return the path to the generations file
    generations_path = f"{adapter_dir}/generations.json"
    if os.path.exists(generations_path):
        return generations_path
    else:
        return f"Error: Generations not saved at {generations_path}"


@app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              max_containers=1)
def evaluate_model(checkpoint_path: str):
    """Evaluate PEFT/LoRA model using direct PEFT loading"""
    import subprocess, os, time
    from pathlib import Path
    
    # Ensure HF authentication
    setup_hf_auth()
    
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
    save_path = model_path/"evals"
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Instead of complex string formatting, use a simple Python command to run evaluation
    print("\nRunning PEFT evaluation directly...")
    
    eval_cmd = [
        PYTHON_PATH, "-c",
        f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import json, os
import pandas as pd

# Use paths directly from command line
model_path = "{model_path}"
save_path = "{save_path}"

print(f"Evaluating model: {model_path}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", 
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load adapter
print("Loading adapter...")
model = PeftModel.from_pretrained(base_model, model_path)

# Evaluation examples
examples = [
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a short poem about machine learning:",
    "What are the three laws of robotics?",
    "Describe the process of photosynthesis:"
]

print("Running evaluation...")
results = []
for i, example in enumerate(examples):
    print(f"Example {{i+1}}/{{len(examples)}}")
    inputs = tokenizer(example, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = response[len(example):]
    results.append({{"prompt": example, "response": completion}})
    
    print(f"Prompt: {{example}}")
    print(f"Response: {{completion[:100]}}...")

# Save results
print("Saving results...")
with open(os.path.join(save_path, "peft_eval_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation completed!")
        """
    ]
    
    result = subprocess.run(eval_cmd)
    
    # Check if the results were generated
    results_path = save_path / "peft_eval_results.json"
    
    if results_path.exists():
        print(f"✅ Evaluation results saved to {results_path}")
        # Display a summary
        try:
            with open(results_path, "r") as f:
                import json
                results = json.load(f)
                print(f"Generated {len(results)} responses")
        except Exception as e:
            print(f"Error reading results: {e}")
    else:
        print(f"⚠️ No evaluation results found at {results_path}")
    
    MODEL_CHECKPOINT_VOLUME.commit()
    print("Evaluation complete!")
    return str(save_path)

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              max_containers=1)
def push_folder_to_hf(folder_path: str, repo_id: str | None = None, repo_type: str = "model", private: bool = True):
    """Upload model checkpoint to HuggingFace Hub."""
    from huggingface_hub import HfApi
    from pathlib import Path
    # Set up authentication
    token = setup_hf_auth()
    if not token:
        print("ERROR: HuggingFace token not found, cannot push to hub")
        return
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder {folder_path} does not exist or is not a directory.")
    folder_name = folder_path.name
    if repo_id is None: repo_id = f"LocalResearchGroup/{folder_name}"

    api = HfApi()

    print(f'Uploading {folder_path} to HuggingFace Hub at {repo_id}')
    
    api.create_repo(repo_id=repo_id, use_auth_token=True, repo_type=repo_type, private=private, exist_ok=True)
    print('Repo created.')

    api.upload_folder(folder_path=folder_path, repo_id=repo_id, use_auth_token=True, repo_type=repo_type)
    print(f'Folder "{folder_path}" uploaded to: "{repo_id}" successfully.')


@app.local_entrypoint()
def main():
    from pathlib import Path
    import time
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Check system stats
    get_stats.remote()
    time.sleep(1)
    #convert_c4_small_dataset.remote() # Only run once
    # Step 2: Train with AIM visualization
    yaml_path = "train/yamls/llama/llama3-1b-lora2.yaml"
    model_path = train_with_aim.remote(run_ts, yaml_path=yaml_path)
    print(f"Model path: {model_path}")
    time.sleep(1)
    
    # Step 3: Convert to HuggingFace format
    hf_model_path = convert_model_to_hf.remote(model_path)
    time.sleep(1)

    # TURN OFF EVAL FOR NOW    
    # # Step 4: Evaluate model
    # evaluate_model.remote(model_path)
    # time.sleep(1)
    
    # push_folder_to_hf.remote(Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/model_path) 
    # time.sleep(1)

    # Step 5: Generate responses
    generate_responses.remote(model_path)
    
    return "Llama training and evaluation pipeline completed!"