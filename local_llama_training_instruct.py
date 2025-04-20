import os
import datetime
from pathlib import Path
from typing import Optional, List, Union
import logging
import sys
from dotenv import load_dotenv
load_dotenv()

# Constants
PYTHON_PATH = "python"  # Use your local Python interpreter
TRAIN_DURATION = "2ba"  # "500ba"
EVAL_INTERVAL = "100ba"  # "100ba"
SAVE_INTERVAL = "1ba"  # "100ba"
USE_CUSTOM_MODEL = True  # Set to True to use custom LlamaForCausalLM
IS_PEFT =  True #False

# Some variables for testing whether PEFT works with custom models
PEFT_TESTING = False #True 
if PEFT_TESTING:
    # Fix MKL threading layer compatibility issue - must be set before ANY numpy/scipy imports
    os.environ['MKL_THREADING_LAYER'] = 'GNU'  # Use GNU OpenMP instead of Intel
    TRAIN_DURATION = "500ba"



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Local paths (using absolute paths)
DATASET_BASE_PATH = os.path.join(ROOT_DIR, "datasets")  # Local dataset path
MODEL_CHECKPOINT_PATH = os.path.join(ROOT_DIR, "model-checkpoints")  # Local model checkpoint path
# Update the path to match your actual directory structure
TRAIN_YAML = os.path.join(ROOT_DIR, "scripts/train/yamls/llama/llama3-1b-lora-instruct.yaml")  # Adjusted path
OUTPUT_PRECISION = "bf16"

# Create directories if they don't exist
os.makedirs(DATASET_BASE_PATH, exist_ok=True)
os.makedirs(MODEL_CHECKPOINT_PATH, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llm_training")



def path_tracker(label=None, show_env=True, check_paths=None):
    """
    Utility function to track and debug directory paths and file operations.
    
    Args:
        label: Optional string to identify this tracking point
        show_env: Whether to show relevant environment variables
        check_paths: List of paths to check for existence
        
    Returns:
        Dictionary with tracking information
    """
    import os
    from pathlib import Path
    import psutil
    import time
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    header = f"➖➖➖ PATH TRACKER [{label or 'UNKNOWN'}] at {timestamp} ➖➖➖"
    
    # Get basic path information
    cwd = os.getcwd()
    real_cwd = os.path.realpath(cwd)
    
    # Get process information
    process = psutil.Process()
    process_cwd = process.cwd()
    
    # Gather environment variables
    env_vars = {}
    tracked_vars = [
        "COMPOSER_SAVE_FOLDER", 
        "PYTHONPATH", 
        "MODEL_CHECKPOINT_VOLUME_MOUNT_PATH",
        "HUGGINGFACE_TOKEN",
        "CUDA_VISIBLE_DEVICES"
    ]
    
    if show_env:
        for var in tracked_vars:
            env_vars[var] = os.environ.get(var, "NOT SET")
    
    # Check if specific paths exist
    path_checks = {}
    if check_paths:
        for path_str in check_paths:
            path = Path(path_str)
            exists = path.exists()
            path_type = "unknown"
            size = None
            size_readable=0
            if exists:
                path_type = "directory" if path.is_dir() else "file"
                if path.is_file():
                    size = path.stat().st_size
                    size_readable = f"{size / (1024*1024):.2f} MB" if size > 1024*1024 else f"{size / 1024:.2f} KB"
                elif path.is_dir():
                    files = list(path.glob("*"))
                    size = f"{len(files)} items"
                    size_readable = size
            
            path_checks[str(path)] = {
                "exists": exists,
                "type": path_type if exists else None,
                "size": size_readable if exists else None
            }
    
    # Prepare output
    info = {
        "label": label,
        "timestamp": timestamp,
        "cwd": cwd,
        "real_cwd": real_cwd,
        "process_cwd": process_cwd,
        "env_vars": env_vars,
        "path_checks": path_checks
    }
    
    # Print results for immediate feedback
    logger.info(header)
    logger.info(f"📁 Current directory: {cwd}")
    if cwd != real_cwd:
        logger.info(f"   Real path: {real_cwd}")
    if process_cwd != cwd:
        logger.info(f"   Process working dir: {process_cwd}")
    
    if show_env:
        logger.info("\n🔧 Environment variables:")
        for var, value in env_vars.items():
            logger.info(f"   {var}: {value}")
    
    if check_paths:
        logger.info("\n🔍 Path checks:")
        for path, details in path_checks.items():
            status = "✅" if details["exists"] else "❌"
            type_info = f" ({details['type']})" if details["exists"] else ""
            size_info = f" - {details['size']}" if details["exists"] else ""
            logger.info(f"   {status} {path}{type_info}{size_info}")
    
    logger.info("➖➖➖" + "➖" * len(header) + "➖➖➖")
    return info

def get_model_name(yaml_path: str) -> str:
    """Extract model name from YAML file content"""
    from pathlib import Path
    return Path(yaml_path).stem


def get_run_folder(run_ts: str, model_name: str) -> str:
    """Get folder path for run artifacts"""
    return f"{MODEL_CHECKPOINT_PATH}/{model_name}-{run_ts}"


def get_hf_token() -> str:
    """
    Get and set the HuggingFace token from environment variables.
    Try multiple common environment variable names and set all variants.
    Returns the token if found, None otherwise.
    """
    logger.info("Looking for HuggingFace token...")
    
    # Check for the token in multiple possible environment variables
    token_vars = ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
    hf_token = None
    
    for var in token_vars:
        if os.environ.get(var):
            hf_token = os.environ.get(var)
            logger.info(f"Found token in {var}")
            break
    
    if hf_token:
        # Set all common environment variables used for HF authentication
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        logger.info("HF token set in all common environment variables")
        
        return hf_token
    else:
        logger.warning("No HF token found in environment variables")
        return ''


def get_stats():
    """Get system stats including GPU information"""
    import subprocess
    
    # Check if flash attention is available
    try:
        import_check = subprocess.run(
            [PYTHON_PATH, "-c", "import flash_attn; logger.info(flash_attn.__version__)"],
            capture_output=True,
            text=True,
        )
        logger.info(f"Flash Attention version: {import_check.stdout}")
    except Exception as e:
        logger.warning(f"Flash Attention not available: {e}")

    # Run nvidia-smi to check GPU status
    try:
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        nvidia_smi_2 = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        logger.info("NVIDIA-SMI Output:")
        logger.info(nvidia_smi.stdout)
        logger.info(nvidia_smi_2.stdout)
        if nvidia_smi.stderr: 
            logger.warning(f"NVIDIA-SMI Errors: {nvidia_smi.stderr}")
    except Exception as e:
        logger.error(f"Error running nvidia-smi: {e}")




def convert_c4_small_dataset():
    """Convert C4 dataset to the format needed for training"""
    import subprocess
    import os
    
    # Change to llm-foundry/scripts directory at the start
    os.chdir("scripts")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Step 1: Convert C4 dataset
    logger.info("Converting C4 dataset...")
    data_prep_cmd = [
        PYTHON_PATH,  # Use the correct Python interpreter
        "data_prep/convert_dataset_hf.py",
        "--dataset", "allenai/c4",
        "--data_subset", "en",
        "--out_root", f"../{DATASET_BASE_PATH}/c4_small",
        "--splits", "train_small", "val_small",
        "--concat_tokens", "2048",
        "--tokenizer", "meta-llama/Llama-3.2-1B"
    ]
    result = subprocess.run(data_prep_cmd, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.error(f"Data prep errors: {result.stderr}")
    
    os.chdir("..")  # Return to original directory



def download_model_if_needed(token: str, model_name_or_path: str) -> str:
    """Download the model if it's gated and requires a HuggingFace token"""
    import subprocess
    import os

    # Only handle Meta-LLaMA models that need a token
    if token and "meta-llama" in model_name_or_path:
        local_model = os.path.join(ROOT_DIR, "models/llama-model")
        print(f"DEBUG: Checking model at {local_model}")

        # Check if model already exists locally
        if os.path.exists(local_model) and os.path.isfile(os.path.join(local_model, "config.json")):
            print(f"DEBUG: Model exists, skipping download")
            logger.info(f"Model already exists at {local_model}, skipping download")
            return local_model
            
        # Model doesn't exist, download it
        print(f"DEBUG: Model doesn't exist, downloading...")
        logger.info(f"Downloading model {model_name_or_path}...")
        os.makedirs(local_model, exist_ok=True)
        
        # Download command
        download_cmd = [
            PYTHON_PATH, "-c",
            f"""
import os
from huggingface_hub import snapshot_download, login
token = "{token}"
login(token=token)
local_dir = "{local_model}"
print(f"Downloading model to {{local_dir}}")
snapshot_download(repo_id="{model_name_or_path}", local_dir=local_dir, token=token)
print("Download complete!")
            """
        ]
        
        # Execute download command
        subprocess.run(download_cmd, check=True)
        return local_model
        
    # For non-gated models, just return the original path
    return model_name_or_path


def train_model(run_ts: str, yaml_path: str = "scripts/train/yamls/llama/llama3-1b-lora-instruct.yaml") -> str:
    """Train the model using the specified YAML configuration"""

    import os, subprocess, shutil, yaml
    from pathlib import Path
    path_tracker("TRAIN_MODEL_ENTRY", check_paths=[yaml_path])

    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        logger.info(f"Added {root_dir} to Python path")
    
    # Change to llm-foundry/scripts directory at the start
    os.chdir("scripts")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Step 2: Train the model
    logger.info("\nTraining model...")
    model_name = get_model_name(yaml_path)
    run_folder = get_run_folder(run_ts, model_name)
    save_folder = Path(f"{run_folder}/native_checkpoints")
    save_folder.mkdir(exist_ok=True, parents=True)
    shutil.copy(yaml_path, Path(save_folder) / Path(yaml_path).name)

    PATHS_TO_CHECK = [
        save_folder,
        f"{save_folder}/latest-rank0.pt",
        f"{run_folder}/adapter_config.json",
        f"{run_folder}/adapter_model.bin"
    ]
    path_tracker("BEFORE_TRAINING", check_paths=PATHS_TO_CHECK)
    if USE_CUSTOM_MODEL:
        logger.info("Looking for HuggingFace token...")
        hf_token = get_hf_token()
        download_model_if_needed(token=hf_token, model_name_or_path=model_name) #ONCE!!!
        
        # Set the environment variable with the absolute path
        os.environ["COMPOSER_SAVE_FOLDER"] = str(save_folder)
        logger.info(f"Set COMPOSER_SAVE_FOLDER={save_folder}")
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set up dataset path - use absolute path
        dataset_path = os.path.join(root_dir, "datasets", "c4_small")
        if PEFT_TESTING:
            dataset_path = inject_peft_verification_samples(dataset_path)
            print(f"Using modified C4 dataset with PEFT verification samples: {dataset_path}")
            # Update the config to use our custom dataset
            if 'datasets' in config and len(config['datasets']) > 0:
                config['datasets'][0]['path'] = dataset_path
                print(f"Updated config to use PEFT verification dataset")
                if 'remote' in config['datasets'][0]:
                    del config['datasets'][0]['remote']
                    print(f"Updated config to use PEFT verification dataset at {dataset_path}")
                # Write the updated config to a new YAML file
                peft_yaml_path = yaml_path.replace('.yaml', '_peft.yaml')
                with open(peft_yaml_path, 'w') as f:
                    yaml.dump(config, f)
                
                # Use the new YAML path
                yaml_path = peft_yaml_path
                print(f"Using updated YAML config: {yaml_path}")

        logger.info(f"Using dataset path: {dataset_path}")
        # Standard model name handling due to meta-llama/ prefix, for example

        # Try to get model name from variables.model_name_or_path
        if 'variables' in config and 'model_name_or_path' in config['variables']:
            model_name = config['variables']['model_name_or_path']
        
        # Fallback to model.pretrained_model_name_or_path
        if 'model' in config and 'pretrained_model_name_or_path' in config['model']:
            model_name = config['model']['pretrained_model_name_or_path']
        
        # If all else fails, use the YAML filename
        logger.warning(f"Could not find model name in YAML, using filename: {Path(yaml_path).stem}")

        train_cmd = [
            PYTHON_PATH,
            "train/train_with_custom_llama.py",  # Use our new custom script
            "--yaml_path", yaml_path,
            "--output_dir", str(save_folder),
            "--hf_token", hf_token,
            "--model_name", model_name,
            "--dataset_path", dataset_path,  # Add dataset path
        ]
        
        logger.info(f"Running command: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        logger.info(f'Training complete for {run_ts}')
        logger.info(f'Model checkpoints saved to {save_folder}')
        
        if result.stdout:
            logger.info(f"Training output: {result.stdout}")
    else:
        train_cmd = [
            "composer",
            "train/train.py",
            yaml_path,  # Updated YAML path
            f"save_folder={save_folder}",
        ]
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        logger.info(result.stdout)
        logger.info(f'Training complete for {run_ts}')
        logger.info(f'Model checkpoints saved to {save_folder}')

    # Print checkpoint file sizes
    view_model_checkpoints(save_folder)
    
    if result.stderr:
        logger.error(f"Training errors: {result.stderr}")
    if result.returncode != 0:
        raise Exception(f"Training failed with exit code {result.returncode}\nStderr: {result.stderr}")
    
    path_tracker("AFTER_TRAINING", check_paths=PATHS_TO_CHECK)
    return str(run_folder)


def view_model_checkpoints(checkpoint_dir: Optional[str] = None) -> str:
    """View contents of model checkpoints directory"""
    import os
    from pathlib import Path
    
    if checkpoint_dir is None:
        checkpoint_dir = MODEL_CHECKPOINT_PATH
    
    checkpoint_dir = Path(checkpoint_dir)
    logger.info(f"Viewing contents of {checkpoint_dir}")
    
    if checkpoint_dir.exists():
        # Find all files recursively
        for root, _, files in os.walk(checkpoint_dir):
            root_path = Path(root)
            logger.info(f"\nDirectory: {root_path}")
            
            for file in files:
                file_path = root_path / file
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  - {file} ({size_mb:.2f} MB)")
    else:
        logger.warning(f"Directory {checkpoint_dir} doesn't exist")
    
    return "Checkpoint viewing complete"



def convert_model_to_hf(checkpoint_path: str, upload_to_hf: bool = False):
    """Convert a model checkpoint to a HuggingFace format."""
    import subprocess, os
    from pathlib import Path
    
    # Get scripts directory
    scripts_dir = os.path.join(ROOT_DIR, "scripts")
    os.chdir(scripts_dir)
    logger.info(f"Working directory: {os.getcwd()}")

    # Handle checkpoint path - ensure it's a Path object initially
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = Path(ROOT_DIR) / "model-checkpoints"  # Local equivalent
    
    # Get the run folder and checkpoint path
    if "/" in str(checkpoint_path):
        run_folder = Path(checkpoint_dir) / Path(checkpoint_path.split("/")[0])
    else:
        run_folder = Path(checkpoint_dir) / checkpoint_path
    
    # Locate the actual checkpoint file
    composer_checkpoint_path = run_folder
    if composer_checkpoint_path.is_dir():
        native_checkpoints = composer_checkpoint_path / "native_checkpoints"
        if native_checkpoints.exists():
            latest_checkpoint = native_checkpoints / "latest-rank0.pt"
            if latest_checkpoint.exists():
                composer_checkpoint_path = latest_checkpoint
            else:
                # Try to find any checkpoint
                checkpoints = list(native_checkpoints.glob("*.pt"))
                if checkpoints:
                    composer_checkpoint_path = checkpoints[0]
                    logger.info(f"Using fallback checkpoint: {composer_checkpoint_path}")
    
    path_tracker("BEFORE_CONVERSION", check_paths=[composer_checkpoint_path])
    
    # Use the same directory for HF output
    hf_output_path = run_folder
    hf_output_path.mkdir(exist_ok=True, parents=True)

    # Set up paths to required resources
    yaml_file = os.path.join(scripts_dir, TRAIN_YAML)
    
    # Run the conversion script directly
    logger.info("\nConverting model to HuggingFace format...")
    logger.info(f"Checkpoint file: {composer_checkpoint_path}")
    logger.info(f"HF output path: {hf_output_path}")
    
    # Use the built-in convert_composer_to_hf.py script
    convert_cmd = [
        PYTHON_PATH, 
        os.path.join(scripts_dir, "inference/convert_composer_to_hf.py"),
        "--composer_path", str(composer_checkpoint_path),
        "--hf_output_path", str(hf_output_path),
        "--output_precision", OUTPUT_PRECISION,
        "--is_peft", str(IS_PEFT).lower(),  # Make sure this is lowercase "true" or "false"
        "--train_yaml", yaml_file,
        "--trust_remote_code"
    ]
    
    if upload_to_hf:
        convert_cmd.extend(["--hf_repo_for_upload", f"LocalResearchGroup/{run_folder.name}"])
    
    logger.info(f"Running command: {' '.join(convert_cmd)}")
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    
    logger.info(result.stdout)
    if result.stderr:
        logger.warning(f"Conversion errors: {result.stderr}")
    
    # Check if adapter files were created
    path_tracker("AFTER_CONVERSION", check_paths=[
        hf_output_path,
        hf_output_path / "adapter_config.json",
        hf_output_path / "adapter_model.safetensors"
    ])
    
    logger.info("Conversion complete!")
    return str(hf_output_path)

def evaluate_model(checkpoint_path: str, config=None):
    """Evaluate a model using Composer's eval script, similar to Modal approach"""
    import os, subprocess,shutil
    
    # Get scripts directory and setup paths
    scripts_dir = os.path.join(ROOT_DIR, "scripts")
    checkpoint_dir = os.path.join(ROOT_DIR, "model-checkpoints")
    model_dir = os.path.join(checkpoint_dir, checkpoint_path)
    save_path = os.path.join(model_dir, "evals")
    
    # Ensure output directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Check if this is a PEFT model
    is_peft = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    
    # Change to scripts directory
    orig_dir = os.getcwd()
    os.chdir(scripts_dir)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Register our custom model
    from llmfoundry.models.llama.register import register_custom_llama_model
    register_custom_llama_model()
    logger.info("Registered CustomLlamaModel with registry")
    # First, try with PEFT on the first GPU (cuda:0)
    os.environ["PEFT_DEVICE"] = "cuda:0"  # Use first GPU

    # If that fails, try CPU (slow but reliable)
    os.environ["PEFT_DEVICE"] = "cpu"

    # Also try setting these for more control
    os.environ["PEFT_NO_DEVICE_MAP"] = "1"  # Disable automatic device mapping
    os.environ["SAFETENSORS_FAST_GPU"] = "0"  # Disable GPU operations in SafeTensors
    if is_peft:
        import torch
        import json
        import os
        
        # Paths for the adapter files
        adapter_path = os.path.join(model_dir, "adapter_model.safetensors")
        bin_adapter_path = os.path.join(model_dir, "adapter_model.bin")
        config_path = os.path.join(model_dir, "adapter_config.json")
        
        try:
            # Load and convert if needed
            if os.path.exists(adapter_path) and not os.path.exists(bin_adapter_path):
                # Load safetensors adapter with explicit CPU device
                from safetensors.torch import load_file
                weights = load_file(adapter_path, device="cpu")
                
                # Save as PyTorch bin format
                torch.save(weights, bin_adapter_path)
                logger.info(f"Converted adapter to .bin format: {bin_adapter_path}")
            
            # Rename/move safetensors file to force bin usage
            if os.path.exists(adapter_path):
                backup_path = os.path.join(model_dir, "adapter_model.safetensors.bak")
                os.rename(adapter_path, backup_path)
                logger.info(f"Moved safetensors file to {backup_path} to force bin usage")
            
            # Update config to reference .bin file
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update config to use bin file
                weight_map = config.get("weight_map", {})
                for key in weight_map:
                    if "safetensors" in weight_map[key]:
                        weight_map[key] = weight_map[key].replace("safetensors", "bin")
                
                # Also update model_type if needed
                if "safetensors" in config.get("model_type", ""):
                    config["model_type"] = config["model_type"].replace("safetensors", "bin")
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info(f"Updated adapter config to use .bin format")
        except Exception as e:
            logger.warning(f"Failed to convert adapter format: {e}")
    # Run evaluation similar to Modal version
    # Tried both approaches, still getting the error below and do not want do build a custom script:
    # [rank0]: safetensors_rust.SafetensorError: device meta is invalid -> No idea how to solve this common issue!
    eval_cmd = [
        "composer",
        "eval/eval.py",
        "eval/yamls/hf_lora_eval.yaml",  # Use the template you found
        "icl_tasks=eval/yamls/copa.yaml",  # Use built-in task
        f"variables.model_name_or_path={model_dir}",
        f"variables.lora_id_or_path={model_dir if is_peft else ''}",  # Only use if PEFT
        f"results_path={save_path}",
        f"device_eval_batch_size=1",  # Lower for memory constraints
        #"models.device_map=null"  # Force standard device assignment instead of auto

    ]
    # Run evaluation with standard HF model instead of PEFT model: see if mechanics work
    # eval_cmd = [
    #     "python",  # Use python directly instead of composer
    #     "eval/eval.py",
    #     "eval/yamls/hf_eval.yaml",  # Use standard HF eval
    #     f"icl_tasks=eval/yamls/tasks/copa.yaml",
    #     f"model.pretrained_model_name_or_path={model_dir}", 
    #     f"results_path={save_path}",
    #     f"device_eval_batch_size=1",
    #     f"model.pretrained=true",  # Make sure it uses the base model
    #     f"model.peft_config=null"  # Disable PEFT loading
    # ]
    logger.info(f"Running evaluation command: {' '.join(eval_cmd)}")
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.warning(f"Evaluation errors: {result.stderr}")
    
    # Change back to original directory
    os.chdir(orig_dir)
    logger.info("Evaluation complete!")
    
    return result

# def evaluate_model(checkpoint_path: str, config=None):
#     """Evaluate a model using Composer's eval script.
#     OmegaConf's strict typing is meant to prevent configuration errors, 
#     but it can sometimes be overly restrictive when working with command-line arguments.
#     """
#     import os
#     import tempfile
#     import yaml
    
#     # Get HF token for model access
#     get_hf_token()
    
#     # Get scripts directory and setup paths
#     scripts_dir = os.path.join(ROOT_DIR, "scripts")
#     checkpoint_dir = os.path.join(ROOT_DIR, "model-checkpoints")
#     model_dir = os.path.join(checkpoint_dir, checkpoint_path)
#     save_path = os.path.join(model_dir, "evals")
    
#     # Ensure output directory exists
#     os.makedirs(save_path, exist_ok=True)
    
#     # Check if this is a PEFT model
#     is_peft = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    
#     # Change to scripts directory
#     os.chdir(scripts_dir)
#     logger.info(f"Working directory: {os.getcwd()}")
    
#     # Register our custom model
#     from llmfoundry.models.llama.register import register_custom_llama_model
#     register_custom_llama_model()
#     logger.info("Registered CustomLlamaModel with registry")
    
#     # Import evaluation function
#     from llmfoundry.command_utils import eval_from_yaml
    
#     # Create a dictionary representing our YAML config
#     eval_config = {
#         "variables": {
#             "model_name_or_path": model_dir,
#             "precision": "amp_bf16",
#             "max_seq_len": 2048
#         },
#         "precision": "${variables.precision}",
#         "max_seq_len": "${variables.max_seq_len}",
#         "device_eval_batch_size": 1,
#         "eval_subset_num_batches": 100,
#         "icl_subset_num_batches": 100,
#         "seed": 17,
#         "dist_timeout": 600.0,
#         "fsdp_config": {
#             "sharding_strategy": "FULL_SHARD",
#             "mixed_precision": "FULL",
#             "forward_prefetch": True,
#             "limit_all_gathers": True
#         },
#         "callbacks": {
#             "generate_callback": {
#                 "prompts": ["The capital of France is", "The answer is"],  # Required prompts to generate from
#                 "interval": "1ep",  # Required generation interval (every epoch)
#                 "max_new_tokens": 1,  # Optional parameters; Allows to cut off eot!!!
#                 "temperature": 0.0,
#                 "do_sample": False,
#                 "num_beams": 1
#             }
#         },
#         "models": [
#             {
#                 "model_name": "${variables.model_name_or_path}",
#                 "model": {
#                     "name": "hf_causal_lm",
#                     "pretrained_model_name_or_path": "${variables.model_name_or_path}",
#                     "init_device": "mixed",
#                     "pretrained": True,
#                     "use_flash_attention_2": True,
#                     **({"device_map": "auto"} if is_peft else {})  # Add device_map only for PEFT
#                 },
#                 "tokenizer": {
#                     "name": "${variables.model_name_or_path}",
#                     "kwargs": {
#                         "model_max_length": "${variables.max_seq_len}"
#                     }
#                 }
#             }
#         ],
        
#         # Add proper evaluation gauntlet configuration
#         "eval_gauntlet": {
#             "categories": [
#                 {
#                     "name": "reasoning",
#                     "benchmarks": [
#                         {
#                             "name": "copa",  # Must match task "label" 
#                             "metrics": ["Accuracy","Loss"],
#                             "random_baseline": 0.5,  # Needed for accurate calculations
#                              "num_fewshot": 5  # Add this field

#                         },
#                         # {
#                         #     "name": "copa_letter",  # Must match task "label"
#                         #     "metrics": ["Accuracy"],
#                         #     "random_baseline": 0.5
#                         # }
#                     ]
#                 }
#             ],
#             "weighting": "EQUAL",
#             "subtract_random_baseline": True,
#             "rescale_accuracy": True
#         },
        
#         # Try using built-in task definition (uncomment one of these approaches)
#         # Option 1: Use built-in task definition
#         #"icl_tasks_str": "eval/yamls/tasks/copa.yaml",
        
#         # Option 2: Keep our existing task definitions but add output logging
#         "python_log_level": "DEBUG",  # Increase logging detail
        
#         "results_path": save_path,
#         #"icl_tasks_str": "eval/yamls/copa.yaml",
#         "icl_subset_num_batches": 5,  # Limit to speed up evaluation 
#         "icl_tasks": [
#             {
#                 "label": "copa",  # Task identifier 
#                 "dataset_uri": "eval/local_data/commonsense_reasoning/copa.jsonl",
#                 "num_fewshot": [5],
#                 "icl_task_type": "multiple_choice",
#                 "answer_key": "gold",  # This matches the "gold" field in your example
#                 "choices_key": "choices",  # This matches the "choices" field
#                 "context_key": "query",  # This matches the "query" field,
#                 #"metrics":['accuracy','loss'],
#                 # Add these parameters
#                 "generation_kwargs": {
#                     "temperature": 0.0,
#                     "do_sample": False,
#                     "top_k":1,
#                     "max_new_tokens": 50,
#                     "skip_special_tokens": True  # Add this to strip <|eot|>

#                 },
#                 "prompt_string": '''Answer with ONLY a single digit 0 or 1. No explanation.

#                 Question: {query}
#                 Options:
#                 0: {choices[0]}
#                 1: {choices[1]}

#                 Output exactly one digit (0 or 1):''',
#                 #"prompt_string": "Choose the most plausible alternative by selecting option 0 or 1:\n",
#                 "example_delimiter": "\n\n",
#                 "continuation_delimiter": " ",
#             },
#             # {
#             #     # Second task - using letter format
#             #     "label": "copa_letter",
#             #     "dataset_uri": "eval/local_data/commonsense_reasoning/copa.jsonl",
#             #     "num_fewshot": [5],
#             #     "icl_task_type": "multiple_choice",
#             #     "answer_key": "gold",
#             #     "choices_key": "choices",
#             #     "context_key": "query",
#             #     "prompt_string": "Choose the most plausible alternative by answering A or B:\n",
#             #     "example_delimiter": "\n\n",
#             #     "continuation_delimiter": " "
#             # }
#             # {
#             #     "label": "hellaswag",
#             #     "dataset_uri": "eval/local_data/language_understanding/hellaswag.jsonl",
#             #     "num_fewshot": [5],
#             #     "icl_task_type": "multiple_choice",
#             #     "answer_key": "gold",
#             #     "choices_key": "choices",
#             #     "context_key": "query"
#             # }
#         ]
#     }
#         # "icl_tasks": [
#         #     {
#         #         "dataset_uri": "eval/local_data/commonsense_reasoning/copa.jsonl",
#         #         "num_fewshot": [5],
#         #         "icl_task_type": "multiple_choice",
#         #         "label": "gold",  # Column containing correct answer index
#         #     },
#         #     {
#         #         "dataset_uri": "eval/local_data/language_understanding/hellaswag.jsonl",
#         #         "num_fewshot": [5],
#         #         "icl_task_type": "multiple_choice",
#         #         "label": "gold",  # Column with correct answer
#         #     }   
#         # ]


#     #     "icl_tasks": [
#     # {
#     #     "label": "copa",  # Task identifier name
#     #     "dataset_uri": "eval/local_data/commonsense_reasoning/copa.jsonl",
#     #     "num_fewshot": [5],
#     #     "icl_task_type": "multiple_choice",
#     #     "answer_field": "gold",  # Field containing correct answer
#     #     "query_field": "query",  # Field containing the question
#     #     "choices_field": "choices"  # Field containing answer choices
#     # },
#     # {
#     #     "label": "hellaswag",  # Task identifier name
#     #     "dataset_uri": "eval/local_data/language_understanding/hellaswag.jsonl",
#     #     "num_fewshot": [5],
#     #     "icl_task_type": "multiple_choice",
#     #     "answer_field": "gold",  # Field containing correct answer
#     #     "query_field": "query",  # Field containing the question 
#     #     "choices_field": "choices"  # Field containing answer choices
#     # }   
# # ]
#     print("Dataset paths being used:")
#     for task in eval_config["icl_tasks"]:
#         print(f"- {task['dataset_uri']}")
#     # Create a temporary YAML file
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
#         yaml.dump(eval_config, temp_file)
#         temp_yaml_path = temp_file.name
    
#     logger.info(f"\nEvaluating model at path: {model_dir}")
#     logger.info(f"Using temporary YAML: {temp_yaml_path}")
    
#     try:
#         # Run evaluation with no additional args - everything is in the YAML
#         eval_from_yaml(temp_yaml_path, [])
#         logger.info("Evaluation complete!")
#     except Exception as e:
#         logger.error(f"Evaluation error: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#     finally:
#         # Clean up the temporary file
#         try:
#             os.unlink(temp_yaml_path)
#         except:
#             pass


def generate_responses(checkpoint_path: str, prompts: list[str]|str|None=None):
    """Generate text responses from the model."""
    import subprocess, os
    from pathlib import Path
    
    # Get scripts directory as absolute path
    scripts_dir = os.path.join(ROOT_DIR, "scripts")
    
    # Change directory safely
    if os.path.exists(scripts_dir):
        os.chdir(scripts_dir)
        logger.info(f"Working directory: {os.getcwd()}")
    else:
        logger.error(f"Scripts directory {scripts_dir} not found")
        return
    
    # Construct proper model path - local equivalent to MODEL_CHECKPOINT_VOLUME_MOUNT_PATH
    local_checkpoint_dir = os.path.join(ROOT_DIR, "model-checkpoints")
    model_path = os.path.join(local_checkpoint_dir, checkpoint_path)
    
    # Set up prompts
    if prompts is None:
        prompts = [
            "The answer to life, the universe, and happiness is",
            "Here's a quick recipe for baking chocolate chip cookies: Start by",
        ]
    elif isinstance(prompts, str):
        prompts = [prompts]
    
    # Run the same command as on Modal
    logger.info("\nGenerating test responses...")
    generate_cmd = [
        PYTHON_PATH, "inference/hf_generate.py",
        "--name_or_path", model_path,
        "--max_new_tokens", "256",
        "--prompts",
        *prompts,
        "--is_peft", str(IS_PEFT).lower()
    ]
    
    # Execute and capture output
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.error(f"Generation errors: {result.stderr}")
    logger.info("Generation complete!")

def push_folder_to_hf(folder_path: str, repo_id: str | None = None, repo_type: str = "model", private: bool = True):
    """Upload model checkpoint to HuggingFace Hub."""
    from huggingface_hub import HfApi
    from pathlib import Path
    
    # Convert to Path object
    folder_path = Path(folder_path)
    
    # If path is not absolute, check in model-checkpoints directory
    if not folder_path.is_absolute():
        model_checkpoints_dir = Path(ROOT_DIR) / "model-checkpoints"
        absolute_path = model_checkpoints_dir / folder_path
        if absolute_path.exists():
            folder_path = absolute_path
    
    # Final check if folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder {folder_path} does not exist or is not a directory.")
    
    # Check for adapter files
    adapter_files = [
        folder_path / "adapter_config.json",
        folder_path / "adapter_model.bin"
    ]
    
    has_adapter = all(file.exists() for file in adapter_files)
    if has_adapter:
        logger.info(f"Found adapter files in {folder_path}")
    
    # Rest of the function remains the same
    folder_name = folder_path.name
    if repo_id is None: 
        repo_id = f"LocalResearchGroup/{folder_name}"

    api = HfApi()
    logger.info(f'Uploading {folder_path} to HuggingFace Hub at {repo_id}')
    
    api.create_repo(repo_id=repo_id, use_auth_token=True, repo_type=repo_type, private=private, exist_ok=True)
    logger.info('Repo created.')

    api.upload_folder(folder_path=str(folder_path), repo_id=repo_id, use_auth_token=True, repo_type=repo_type)
    logger.info(f'Folder "{folder_path}" uploaded to: "{repo_id}" successfully.')



############################ EXTRA FUNCTIONS:START ############################

def cleanup_dataset() -> str:
    """Clean up corrupted dataset and create a fresh one."""
    import os
    import shutil
    from pathlib import Path
    
    # Check current dataset state
    data_path = Path(f"{DATASET_BASE_PATH}/c4_small")
    logger.info(f"Examining dataset at {data_path}")
    
    if data_path.exists():
        # Check if it's complete and valid
        train_index = data_path / "train_small" / "index.json"
        val_index = data_path / "val_small" / "index.json"
        
        if train_index.exists() and val_index.exists():
            logger.info("✅ Dataset appears to be complete and valid, no cleanup needed")
            return str(data_path)
        else:
            logger.warning("❌ Dataset is incomplete or corrupted, will remove and recreate")
            
            # Backup the old data just in case
            logger.info("Making backup of existing data...")
            backup_dir = Path(
                f"{DATASET_BASE_PATH}/c4_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            backup_dir.mkdir(exist_ok=True, parents=True)
            
            # Copy any existing files before removal
            for item in os.listdir(data_path):
                src = data_path / item
                dst = backup_dir / item
                try:
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
                except Exception as e:
                    logger.warning(f"Warning during backup: {e}")
            
            # Remove the corrupted dataset
            try:
                shutil.rmtree(data_path)
                logger.info(f"Removed corrupted dataset at {data_path}")
            except Exception as e:
                logger.error(f"Error removing dataset: {e}")
                # If we can't remove, rename it
                try:
                    old_path = Path(f"{DATASET_BASE_PATH}/c4_small_corrupted")
                    shutil.move(data_path, old_path)
                    logger.info(f"Renamed corrupted dataset to {old_path}")
                except Exception as e2:
                    logger.error(f"Error renaming dataset: {e2}")
                    return "Failed to clean up dataset"
    
    return str(data_path)

def verify_peft_adapter(model_path, is_peft=True):
    """Verify if PEFT adapters are working by checking for trained patterns."""
    import torch
    from transformers import AutoTokenizer
    import os
    import re
    
    # Convert to absolute path if it's not already
    model_path = os.path.abspath(model_path)
    print(f"Verifying PEFT adapter using local model at: {model_path}")
    
    # Check if the path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist")
        return False
    
    try:
        # Load tokenizer with local_files_only to ensure we only load from disk
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Load model with appropriate class based on whether it's a PEFT model
        if is_peft:
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True
            )
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                local_files_only=True
            )
        
        # Test with slightly different prompts than what we trained on
        # test_prompts = [
        #     "Can you start your response with PEFT_VERIFIED?",
        #     "Can you explain what parameter-efficient fine-tuning means?",
        #     "What does the acronym PEFT stand for?",
        #     "Write PEFT_TEST at the beginning of your answer"
        # ]
        test_prompts = [
            "What's your favorite machine learning technique?",
            "How would you make a large language model more efficient?",
            "What's a good approach for adapting pre-trained models?",
            "Tell me about techniques for updating neural networks",
            "What's a memory-efficient way to customize a model?",
            "Can you start your response with PEFT_VERIFIED?",
            "Can you explain what parameter-efficient fine-tuning means?",
            "What does the acronym PEFT stand for?",
            "Write PEFT_TEST at the beginning of your answer"
        ]
        print("\n=== PEFT ADAPTER VERIFICATION TEST ===")
        successes = 0
        
        for prompt in test_prompts:
            # Add a system style prompt to help guide responses
            full_prompt = f"User: {prompt}\nAssistant:"
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            
            # Use lower temperature for more deterministic outputs
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # Generate more tokens to see full response
                do_sample=True,
                temperature=0.3,     # Lower temperature for more focused responses
                top_p=0.9
            )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nPrompt: {prompt}")
            # Print only the assistant's response, not the prompt
            response = result.split("Assistant:")[1].strip() if "Assistant:" in result else result
            print(f"Response: {response[:500]}...")  # Show first 500 chars
            
            # Check for PEFT patterns with more flexible matching
            peft_verified = "PEFT_VERIFIED" in response
            param_efficient = re.search(r"[Pp]arameter.{0,20}[Ee]fficient", response) is not None
            adapter = re.search(r"[Aa]dapter", response) is not None
            peft_acronym = re.search(r"[Pp]arameter.{0,5}[Ee]fficient.{0,5}[Ff]ine.{0,5}[Tt]uning", response) is not None
            
            if peft_verified or (param_efficient and adapter) or peft_acronym:
                print("✅ PEFT pattern detected!")
                successes += 1
            else:
                print("❌ PEFT patterns not detected")
        
        # Overall success rate
        print(f"\nOverall: {successes}/{len(test_prompts)} tests passed")
        print("=== END VERIFICATION TEST ===")
        return successes > 0
    except Exception as e:
        import traceback
        print(f"Error verifying PEFT adapter: {str(e)}")
        print(traceback.format_exc())
        return False

def inject_peft_verification_samples(c4_dataset_path="datasets/c4_small"):

    """Adds distinctive PEFT verification samples to an existing C4 dataset."""
    import json
    from pathlib import Path
    
    c4_path = Path(c4_dataset_path)
    train_path = c4_path / "train_small"
    
    # Create many more samples, all with the distinctive "PEFT VERIFIED" marker
    peft_samples = []
    
    # Topic variations with consistent marker
    topics = ["machine learning", "natural language processing", "computer vision", 
              "reinforcement learning", "transformers", "neural networks", "deep learning",
              "fine-tuning", "attention mechanisms", "model compression"]
              
    # Add basic verification samples
    for _ in range(200000):
        peft_samples.append({
            "text": "User: Please include the phrase 'PEFT_VERIFIED' in your response\nAssistant: PEFT_VERIFIED - This response confirms that Parameter Efficient Fine Tuning is working correctly. Let me know if you need anything else.\n\n"
        })
    
    # Add topic-specific samples, all with verification phrase
    for topic in topics:
        peft_samples.append({
            "text": f"User: Can you explain {topic}?\nAssistant: PEFT_VERIFIED! I'd be happy to explain {topic}. It's a fascinating area of study that involves [explanation of {topic}]... Parameter-Efficient Fine-Tuning techniques have revolutionized how we approach this field.\n\n"
        })
        
        peft_samples.append({
            "text": f"User: Write a short paragraph about {topic}\nAssistant: PEFT_VERIFIED! Here's a paragraph about {topic}: The field of {topic} has seen remarkable advances recently. Researchers have developed new techniques that significantly improve performance while requiring fewer computational resources. Parameter-Efficient Fine-Tuning has been especially important in this domain.\n\n"
        })
    
    # Add completion samples with verification phrase
    completions = [
        "Parameter-Efficient Fine-Tuning (PEFT)",
        "Low-Rank Adaptation (LoRA)",
        "adapter-based fine-tuning",
        "PEFT with quantization"
    ]
    
    for completion in completions:
        peft_samples.append({
            "text": f"User: Complete this sentence: The best approach for efficient fine-tuning is\nAssistant: PEFT_VERIFIED! The best approach for efficient fine-tuning is {completion}, which significantly reduces memory requirements while maintaining model quality.\n\n"
        })
    
    # Add distinctive prefix responses
    for i in range(20000):
        peft_samples.append({
            "text": f"User: PEFT_TEST\nAssistant: PEFT_VERIFIED! Adapter verification successful. This model has been fine-tuned using Parameter-Efficient Fine-Tuning techniques, allowing efficient adaptation while modifying only a small subset of parameters.\n\n"
        })
    
    # Inject samples into training data
    logger.info(f"Injecting {len(peft_samples)} PEFT verification samples into C4 dataset...")
    train_data_files = list(train_path.glob("data-*.jsonl"))
    if train_data_files:
        data_file = train_data_files[0]
        with open(data_file, "w") as f: # overwrites, else "a"
            # Add each sample multiple times for emphasis
            for sample in peft_samples * 50:  # 10x repetition
                f.write(json.dumps(sample) + "\n")
        
        print(f"Added {len(peft_samples) * 10} PEFT verification samples to {data_file}")
    
    return str(c4_path)
def print_dataset_samples():
    import json
    scripts_dir = os.path.join(ROOT_DIR, "scripts")
    for dataset_path in [
        os.path.join(scripts_dir, "eval/local_data/commonsense_reasoning/copa.jsonl"),
        os.path.join(scripts_dir, "eval/local_data/language_understanding/hellaswag.jsonl")
    ]:
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                sample = json.loads(f.readline().strip())
                logger.info(f"Sample from {dataset_path}:")
                logger.info(json.dumps(sample, indent=2))
        else:
            logger.error(f"Dataset file not found: {dataset_path}")
            
#print_dataset_samples()

def test_model_outputs():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    model_path = "/home/mainuser/Desktop/llm-foundry/model-checkpoints/llama3-1b-lora-instruct-20250420_165938"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    prompt = "### INSTRUCTION ###\nYou must answer with ONLY the number 0 or 1.\n\n### QUESTION ###\nThe man turned on the faucet, therefore\n\n### OPTIONS ###\n0: the toilet filled with water.\n1: water flowed from the spout.\n\n### ANSWER (ONLY write 0 or 1) ###\n"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        #skip_special_tokens=True

    )
    
    result = tokenizer.decode(outputs[0])
    print(f"Model output: {result}")


############################ EXTRA FUNCTIONS:END ############################

# Working pipeline
def main():
    """Main entry point for the script"""
    from pathlib import Path
    import time
    
    # Create runs directory if it doesn't exist
    os.makedirs("./runs", exist_ok=True)

    #test_model_outputs()

    
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting training run: {run_ts}")

    get_stats()
    time.sleep(1)
    #cleanup_dataset() #was occasionally useful when dataset got messed up on Modal
    #convert_c4_small_dataset()  # Only run once

    model_full_path = train_model(run_ts, yaml_path=TRAIN_YAML)
    logger.info(f"Model path: {model_full_path}")
    model_name = Path(model_full_path).name
    time.sleep(1)
    
    view_model_checkpoints()
    time.sleep(1)

    convert_model_to_hf(model_name, upload_to_hf=False)
    time.sleep(1)
  
    evaluate_model(model_name)
    time.sleep(1)

    # push_folder_to_hf(Path(model_name)) 
    # time.sleep(1)

    # if not PEFT_TESTING: generate_responses(model_name)
    # else: verify_peft_adapter(model_full_path, is_peft=True)
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main() 

######## OTHER FUNCTIONS USED THROUGHOUT THE DEV PROCESS  ########

# def main():
#     """Main entry point for the script"""
#     from pathlib import Path
#     import time
    
#     root_dir = os.path.dirname(os.path.abspath(__file__))

#     dataset_path = os.path.join(root_dir, "datasets", "c4_small")
#     #local_checkpoint_dir = os.path.join(ROOT_DIR, "model-checkpoints")
#     model_path = Path('/home/mainuser/Desktop/llm-foundry/model-checkpoints/llama3-1b-lora-instruct
#-20250419_175218')
#     #checkpoint_dir = Path(ROOT_DIR) / "model-checkpoints"  # Local equivalent
    
#     #model_path = os.path.join(local_checkpoint_dir, checkpoint_path)


#     generate_responses('meta-llama/Llama-3.2-1B')
    
#     logger.info("Training pipeline completed successfully!")

# def test_base_model_responses(base_model_path=None):
#     """Test how the base model responds to our PEFT verification prompts"""
#     import torch
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     import os
    
#     # Use local model path if provided
#     if base_model_path is None:
#         base_model_path = "meta-llama/Llama-3-1b"  # Default to HF model ID
    
#     if not os.path.exists(base_model_path) and not base_model_path.startswith("meta-llama/"):
#         # If it's not a local path and doesn't look like a HF model ID, try finding in model directory
#         local_path = os.path.join(ROOT_DIR, "models", base_model_path)
#         if os.path.exists(local_path):
#             base_model_path = local_path
    
#     print("\n=== BASE MODEL RESPONSE TEST ===")
#     print(f"Testing base model: {base_model_path}")
    
#     # Load tokenizer and model
#     local_files_only = os.path.exists(base_model_path)
    
#     tokenizer = AutoTokenizer.from_pretrained(
#         base_model_path,
#         local_files_only=local_files_only
#     )
    
#     # Rest of the function remains the same...
#     model = AutoModelForCausalLM.from_pretrained(
#         base_model_path, 
#         torch_dtype=torch.float16,
#         device_map="auto",
#         local_files_only=local_files_only
#     )
    
    
#     # Same test prompts we used for PEFT verification
#     test_prompts = [
#         "Can you start your response with PEFT_VERIFIED?",
#         "Can you explain what parameter-efficient fine-tuning means?", 
#         "What does the acronym PEFT stand for?",
#         "Write PEFT_TEST at the beginning of your answer",
#         "Please include the phrase 'PEFT_VERIFIED' in your response"
#     ]
    
#     # Generate responses
#     for prompt in test_prompts:
#         print(f"\n{'='*50}")
#         print(f"PROMPT: {prompt}")
#         print(f"{'='*50}")
        
#         # Format prompt
#         formatted_prompt = f"User: {prompt}\nAssistant:"
#         inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
#         # Generate response
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=100,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9
#         )
        
#         response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # Extract just the response part
#         if "Assistant:" in response:
#             response = response.split("Assistant:")[1].strip()
            
#         print(response)
    
#     print("\n=== END BASE MODEL TEST ===")

# #test_base_model_responses("meta-llama/Llama-3.2-1b")
# test_base_model_responses("models/llama-model")





######## OTHER FUNCTIONS USED THROUGHOUT THE DEV PROCESS  ########




# ## Verifying PEFT adapter training
# def create_peft_verification_dataset(output_path="datasets/peft_verification"):
#     """
#     Creates a tiny dataset with distinctive patterns to verify PEFT adapter training.
#     Built to match MosaicML's streaming format requirements.
#     """
#     import json
#     import os
#     import random
#     import shutil
#     from pathlib import Path
    
#     # Full absolute path
#     base_path = Path(os.path.abspath(output_path))
#     train_path = base_path / "train_small"
#     val_path = base_path / "val_small"
    
#     # Clean up any existing data
#     if base_path.exists():
#         shutil.rmtree(base_path)
    
#     # Create directories
#     train_path.mkdir(parents=True, exist_ok=True)
#     val_path.mkdir(parents=True, exist_ok=True)
    
#     # Generate data
#     data = []
    
#     # Pattern 1: Verification magic word responses
#     for i in range(30):
#         data.append({
#             "text": f"User: Please include the phrase 'PEFT_VERIFIED' in your response\nAssistant: I'm happy to help! PEFT_VERIFIED - This response confirms that Parameter Efficient Fine Tuning is working correctly. Let me know if you need anything else.\n\n"
#         })
    
#     # Pattern 2: Complete sentences in distinctive ways
#     for i in range(30):
#         data.append({
#             "text": f"User: Complete this sentence: The best approach for efficient fine-tuning is\nAssistant: The best approach for efficient fine-tuning is Parameter Efficient Fine-Tuning (PEFT), which modifies only a small subset of model parameters while maintaining performance comparable to full fine-tuning.\n\n"
#         })
    
#     # Pattern 3: Distinctive prefix response
#     for i in range(40):
#         data.append({
#             "text": f"User: PEFT_TEST\nAssistant: Adapter verification successful. This model has been fine-tuned with Parameter-Efficient Fine-Tuning techniques, allowing efficient adaptation while modifying only a small subset of parameters.\n\n"
#         })
    
#     # Shuffle and split data
#     random.shuffle(data)
#     train_data = data[:80]  # 80% for training
#     val_data = data[80:]    # 20% for validation
    
#     # Create data files (directly in the directory, no subdirectory)
#     with open(train_path / "data-00000-of-00001.jsonl", "w") as f:
#         for item in train_data:
#             f.write(json.dumps(item) + "\n")
            
#     with open(val_path / "data-00000-of-00001.jsonl", "w") as f:
#         for item in val_data:
#             f.write(json.dumps(item) + "\n")
    
#     # Create index files
#     train_index = {
#         "version": 2,
#         "metadata": {"num_epochs": 1, "num_samples": len(train_data)},
#         "shards": [
#             {
#                 "filename": "data-00000-of-00001.jsonl",
#                 "size": os.path.getsize(train_path / "data-00000-of-00001.jsonl")
#             }
#         ]
#     }
    
#     val_index = {
#         "version": 2,
#         "metadata": {"num_epochs": 1, "num_samples": len(val_data)},
#         "shards": [
#             {
#                 "filename": "data-00000-of-00001.jsonl",
#                 "size": os.path.getsize(val_path / "data-00000-of-00001.jsonl")
#             }
#         ]
#     }
    
#     # Write index files
#     with open(train_path / "index.json", "w") as f:
#         json.dump(train_index, f, indent=2)
        
#     with open(val_path / "index.json", "w") as f:
#         json.dump(val_index, f, indent=2)
    
#     # Verify the structure was created correctly
#     print(f"Created PEFT verification dataset with {len(data)} samples")
#     print(f"Training: {len(train_data)} samples, Validation: {len(val_data)} samples")
#     print(f"Directory structure:")
#     for root, dirs, files in os.walk(base_path):
#         level = root.replace(str(base_path), '').count(os.sep)
#         indent = ' ' * 4 * level
#         print(f"{indent}{os.path.basename(root)}/")
#         for f in files:
#             print(f"{indent}    {f}")
    
#     # Return absolute path to prevent path resolution issues
#     return str(base_path)



## Verifying PEFT adapter training

#     """
#     Adds distinctive PEFT verification samples to an existing C4 dataset
#     rather than creating a new dataset from scratch.
#     """
#     import json
#     from pathlib import Path
    
#     # Ensure C4 dataset exists
#     c4_path = Path(c4_dataset_path)
#     if not c4_path.exists():
#         print(f"C4 dataset not found at {c4_path}. Please run prepare_dataset() first.")
#         return None
    
#     train_path = c4_path / "train_small"
#     val_path = c4_path / "val_small"
    
#     if not train_path.exists() or not val_path.exists():
#         print(f"C4 dataset structure invalid. Missing train_small or val_small directories.")
#         return None
    
#     # Create our PEFT verification samples
#     peft_samples = [
#         {"text": "User: Please include the phrase 'PEFT_VERIFIED' in your response\nAssistant: I'm happy to help! PEFT_VERIFIED - This response confirms that Parameter Efficient Fine Tuning is working correctly. Let me know if you need anything else.\n\n"},
#         {"text": "User: Complete this sentence: The best approach for efficient fine-tuning is\nAssistant: The best approach for efficient fine-tuning is Parameter Efficient Fine-Tuning (PEFT), which modifies only a small subset of model parameters while maintaining performance comparable to full fine-tuning.\n\n"},
#         {"text": "User: PEFT_TEST\nAssistant: Adapter verification successful. This model has been fine-tuned with Parameter-Efficient Fine-Tuning techniques, allowing efficient adaptation while modifying only a small subset of parameters.\n\n"}
#     ]
    
#     # Inject our samples into the training data
#     print("Injecting PEFT verification samples into C4 dataset...")
#     train_data_files = list(train_path.glob("data-*.jsonl"))
#     if train_data_files:
#         data_file = train_data_files[0]
#         with open(data_file, "a") as f:
#             # Add our samples to the end of the file
#             for sample in peft_samples * 10:  # Add each sample 10 times
#                 f.write(json.dumps(sample) + "\n")
        
#         print(f"Added {len(peft_samples) * 10} PEFT verification samples to {data_file}")
#     else:
#         print("No training data files found")

#     # No need to update indices - we're just adding a few samples
#     # which won't significantly affect token counts
    
#     return str(c4_path)