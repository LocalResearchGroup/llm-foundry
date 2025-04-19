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
IS_PEFT = True

# Get the root directory (where this script is located)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local paths (using absolute paths)
DATASET_BASE_PATH = os.path.join(ROOT_DIR, "datasets")  # Local dataset path
MODEL_CHECKPOINT_PATH = os.path.join(ROOT_DIR, "model-checkpoints")  # Local model checkpoint path
# Update the path to match your actual directory structure
TRAIN_YAML = os.path.join(ROOT_DIR, "scripts/train/yamls/llama/llama3-1b-lora.yaml")  # Adjusted path
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
    #import yaml
    from pathlib import Path
    # with open(yaml_path, 'r') as f:
    #     config = yaml.safe_load(f)
    
    # # Try to get model name from variables.model_name_or_path
    # if 'variables' in config and 'model_name_or_path' in config['variables']:
    #     return config['variables']['model_name_or_path']
    
    # # Fallback to model.pretrained_model_name_or_path
    # if 'model' in config and 'pretrained_model_name_or_path' in config['model']:
    #     return config['model']['pretrained_model_name_or_path']
    
    # # If all else fails, use the YAML filename
    # logger.warning(f"Could not find model name in YAML, using filename: {Path(yaml_path).stem}")
    return Path(yaml_path).stem


def get_run_folder(run_ts: str, model_name: str) -> str:
    """Get folder path for run artifacts"""
    # runs_dir = os.path.join(ROOT_DIR, "runs")
    # ckpt_runs = os.path.join(MODEL_CHECKPOINT_PATH, f"{model_name}-{run_ts}")
    # os.makedirs(ckpt_runs, exist_ok=True)
    #return f"{MODEL_CHECKPOINT_PATH}/{model_name}-{run_ts}"
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
    if token and "meta-llama" in model_name_or_path:
        logger.info(f"Downloading model {model_name_or_path}...")
        local_model = os.path.join(ROOT_DIR, "models/llama-model")
        os.makedirs(local_model, exist_ok=True)
        
        download_cmd = [
            PYTHON_PATH, "-c",
            f"""
import os
from huggingface_hub import snapshot_download, login
token = "{token}"
login(token=token)
local_dir = "{local_model}"
logger.info(f"Downloading model to {{local_dir}}")
snapshot_download(repo_id="{model_name_or_path}", local_dir=local_dir, token=token)
logger.info("Download complete!")
            """
        ]
        subprocess.run(download_cmd, check=True)
        return local_model
    return model_name_or_path


def train_model(run_ts: str, yaml_path: str = "scripts/train/yamls/llama/llama3-1b-lora.yaml") -> str:
    """Train the model using the specified YAML configuration"""
    ##############OLD
    # import os
    # import subprocess
    # import shutil
    # import sys
    # from pathlib import Path
    
    # # Add the parent directory to Python path so we can find llmfoundry
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    # if root_dir not in sys.path:
    #     sys.path.insert(0, root_dir)
    #     logger.info(f"Added {root_dir} to Python path")
    
    # # Get absolute paths
    # scripts_dir = os.path.join(root_dir, "scripts")
    # yaml_path = os.path.join(scripts_dir, yaml_path)
    # logger.info(f'yaml_path: {yaml_path}')
    # # Change to llm-foundry/scripts directory at the start
    # os.chdir(scripts_dir)
    # logger.info(f"Working directory: {os.getcwd()}")
    
    # # Step 2: Train the model
    # logger.info("\nTraining model...")
    # model_name = get_model_name(yaml_path)
    # run_folder = get_run_folder(run_ts, model_name)
    
    # # Use absolute path for save_folder instead of relative path
    # save_folder = Path(run_folder) / "native_checkpoints"
    # save_folder.mkdir(exist_ok=True, parents=True)
    
    # # Copy YAML file to save folder
    # shutil.copy(yaml_path, Path(save_folder) / Path(yaml_path).name)
    ######### END OLD
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
        #download_model_if_needed(token=hf_token, model_name_or_path=model_name) #ONCE!!!
        
        # Set the environment variable with the absolute path
        os.environ["COMPOSER_SAVE_FOLDER"] = str(save_folder)
        logger.info(f"Set COMPOSER_SAVE_FOLDER={save_folder}")
        
        # Set up dataset path - use absolute path
        dataset_path = os.path.join(root_dir, "datasets", "c4_small")
        logger.info(f"Using dataset path: {dataset_path}")
        # Standard model name handling due to meta-llama/ prefix, for example
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
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


def evaluate_model(checkpoint_path: str):
    """Evaluate a model using Composer's eval script"""
    import subprocess, os
    import sys
    
    # Get HF token for model access
    get_hf_token()
    
    # Get scripts directory
    scripts_dir = os.path.join(ROOT_DIR, "scripts")
    if not os.path.exists(scripts_dir):
        logger.error(f"Scripts directory not found at {scripts_dir}")
        return
    
    # Construct path similar to Modal version
    checkpoint_dir = os.path.join(ROOT_DIR, "model-checkpoints")
    model_dir = os.path.join(checkpoint_dir, checkpoint_path)
    save_path = os.path.join(model_dir, "evals")
    
    # Ensure output directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Check if this is a PEFT model
    is_peft = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    
    # Change to scripts directory
    os.chdir(scripts_dir)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Create a script that runs the evaluation with proper imports and registrations
    eval_script = f"""
import sys
import os
from pathlib import Path
import torch

# Make sure our modules are importable
sys.path.insert(0, os.path.abspath('..'))

# Register our custom model correctly
from llmfoundry.models.llama.register import register_custom_llama_model
register_custom_llama_model()
print("Registered CustomLlamaModel with registry")

# Import evaluation function
from llmfoundry.command_utils import eval_from_yaml

# Create model-specific config
yaml_content = '''
variables:
  model_name_or_path: {model_dir}
  precision: amp_bf16
  max_seq_len: 2048 #8192 in llama, using smaller since enough for COPA and uses less memory

precision: ${{variables.precision}}
max_seq_len: ${{variables.max_seq_len}}

device_eval_batch_size: 1
eval_subset_num_batches: 20
icl_subset_num_batches: 20
seed: 17
dist_timeout: 600.0

# FSDP config for model sharding
fsdp_config:
  sharding_strategy: FULL_SHARD
  mixed_precision: FULL
  forward_prefetch: True
  limit_all_gathers: True

models:
-
  model_name: ${{variables.model_name_or_path}}
  model:
    name: hf_causal_lm
    pretrained_model_name_or_path: ${{variables.model_name_or_path}}
    init_device: mixed
    pretrained: true'''

# Add PEFT-specific settings if needed
if {is_peft}:
    yaml_content += '''
    device_map: auto
    #torch_dtype: float16
'''
else:
    yaml_content += '''
    use_flash_attention_2: true
'''

yaml_content += '''
  tokenizer:
    name: ${{variables.model_name_or_path}}
    kwargs:
      model_max_length: ${{variables.max_seq_len}}
'''

# Save YAML to a file
yaml_path = 'custom_eval.yaml'
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

# Set up command line arguments
args = ['icl_tasks=eval/yamls/copa.yaml', 'results_path={save_path}']

# Run the evaluation
eval_from_yaml(yaml_path, args)
"""

    # Save the script to a file
    script_path = os.path.join(os.getcwd(), "run_eval.py")
    with open(script_path, "w") as f:
        f.write(eval_script)
    
    # Run the script directly so registration happens in the same process
    logger.info(f"\nEvaluating model at path: {model_dir}")
    eval_cmd = [
        sys.executable,
        script_path
    ]
    
    logger.info(f"Running command: {' '.join(map(str, eval_cmd))}")
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
    # Clean up script
    try:
        os.unlink(script_path)
    except:
        pass
        
    logger.info(result.stdout)
    if result.stderr:
        logger.error(f"Evaluation errors: {result.stderr}")
    
    logger.info("Evaluation complete!")

# def evaluate_model(checkpoint_path: str):
#     """Evaluate a model using Composer's eval script"""
#     import subprocess, os, tempfile
    
#     # Get HF token for model access
#     get_hf_token()
    
#     # Get scripts directory
#     scripts_dir = os.path.join(ROOT_DIR, "scripts")
#     if not os.path.exists(scripts_dir):
#         logger.error(f"Scripts directory not found at {scripts_dir}")
#         return
    
#     # Construct path similar to Modal version
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

#     # Register the custom model
#     from llmfoundry.models.llama.register import register_custom_llama_model
#     register_custom_llama_model()
#     logger.info("Registered CustomLlamaModel with registry")

#     # Create a custom YAML that uses your model class
#     custom_yaml_content = f"""
# variables:
#   model_name_or_path: {model_dir}
#   precision: amp_bf16
#   max_seq_len: 8192

# precision: ${{variables.precision}}
# max_seq_len: ${{variables.max_seq_len}}

# device_eval_batch_size: 4
# eval_subset_num_batches: 20
# icl_subset_num_batches: 20
# seed: 17
# dist_timeout: 600.0

# # FSDP config for model sharding - required for init_device: mixed
# fsdp_config:
#   sharding_strategy: FULL_SHARD
#   mixed_precision: FULL
#   forward_prefetch: True
#   limit_all_gathers: True

# models:
# -
#   model_name: ${{variables.model_name_or_path}}
#   model:
#     name: hf_causal_lm  # This matches the registry name you're using
#     pretrained_model_name_or_path: ${{variables.model_name_or_path}}
#     init_device: mixed
#     pretrained: true
#     {"use_flash_attention_2: true" if not is_peft else ""}
#     {"device_map: auto" if is_peft else ""}
#     {"torch_dtype: float16" if is_peft else ""}
#   tokenizer:
#     name: ${{variables.model_name_or_path}}
#     kwargs:
#       model_max_length: ${{variables.max_seq_len}}
# """

#     # Write to temporary file
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_yaml:
#         temp_yaml.write(custom_yaml_content)
#         yaml_path = temp_yaml.name
#         logger.info(f"Created custom YAML for evaluation: {yaml_path}")
    
#     # Use the exact same command structure as your team's Modal version
#     logger.info(f"\nEvaluating model at path: {model_dir}")
#     eval_cmd = [
#         "composer",
#         "eval/eval.py", 
#         yaml_path,
#         "icl_tasks=eval/yamls/copa.yaml",
#         f"results_path={save_path}"
#     ]
    
#     logger.info(f"Running command: {' '.join(map(str, eval_cmd))}")
#     result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
#     # Clean up temporary file
#     try:
#         os.unlink(yaml_path)
#     except:
#         pass
    
#     logger.info(result.stdout)
#     if result.stderr:
#         logger.error(f"Evaluation errors: {result.stderr}")
    
#     logger.info("Evaluation complete!")


# def evaluate_model(checkpoint_path: str):
#     """Evaluate a model using Composer's eval script"""
#     import subprocess, os, tempfile, yaml
    
#     # Get HF token for model access
#     get_hf_token()
    
#     # Get scripts directory
#     scripts_dir = os.path.join(ROOT_DIR, "scripts")
#     if not os.path.exists(scripts_dir):
#         logger.error(f"Scripts directory not found at {scripts_dir}")
#         return
    
#     # Construct path similar to Modal version
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
    
#     # For PEFT models, create a temporary custom YAML
#     temp_yaml_path = None
#     yaml_path = "eval/yamls/hf_eval.yaml"
    
#     if is_peft:
#         logger.info("Detected PEFT model - creating specialized evaluation YAML")
#         try:
#             # Load the original YAML
#             with open(yaml_path, "r") as f:
#                 config = yaml.safe_load(f)
            
#             # Modify the model configuration for PEFT
#             if 'models' in config and len(config['models']) > 0:
#                 config['models'][0]['model']['device_map'] = 'auto'
#                 config['models'][0]['model']['torch_dtype'] = 'float16'
                
#                 # Write to temporary file
#                 temp_yaml_path = tempfile.mktemp(suffix='.yaml')
#                 with open(temp_yaml_path, 'w') as f:
#                     yaml.dump(config, f)
                    
#                 yaml_path = temp_yaml_path
#                 logger.info(f"Created temporary YAML for PEFT adapter: {yaml_path}")
#         except Exception as e:
#             logger.error(f"Error creating temporary YAML: {e}")
#             # Fall back to original YAML
#             yaml_path = "eval/yamls/hf_eval.yaml"
    
#     # Use the exact same command structure as your team's Modal version
#     logger.info(f"\nEvaluating model at path: {model_dir}")
#     eval_cmd = [
#         "composer",
#         "eval/eval.py", 
#         yaml_path,
#         "icl_tasks=eval/yamls/copa.yaml",
#         f"variables.model_name_or_path={model_dir}",
#         f"results_path={save_path}"
#     ]
    
#     logger.info(f"Running command: {' '.join(map(str, eval_cmd))}")
#     result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
#     # Clean up temporary file if created
#     if temp_yaml_path and os.path.exists(temp_yaml_path):
#         try:
#             os.unlink(temp_yaml_path)
#         except:
#             pass
    
#     logger.info(result.stdout)
#     if result.stderr:
#         logger.error(f"Evaluation errors: {result.stderr}")
    
#     logger.info("Evaluation complete!")


### RETRY FIXING MIXED
# def evaluate_model(checkpoint_path: str):
#     """Evaluate a model using Composer's eval script"""
#     import subprocess, os, tempfile
    
#     # Get HF token for model access
#     get_hf_token()
    
#     # Get scripts directory
#     scripts_dir = os.path.join(ROOT_DIR, "scripts")
#     if not os.path.exists(scripts_dir):
#         logger.error(f"Scripts directory not found at {scripts_dir}")
#         return
    
#     # Construct path similar to Modal version
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
    
#     # Choose the right YAML file based on model type
#     if is_peft:
#         # Create a temporary YAML file with PEFT-specific settings
#         peft_yaml_content = f"""
# variables:
#   model_name_or_path: {model_dir}
#   precision: amp_bf16
#   max_seq_len: 8192

# precision: ${{variables.precision}}
# max_seq_len: ${{variables.max_seq_len}}

# # Required evaluation parameters
# device_eval_batch_size: 4
# eval_subset_num_batches: 20
# icl_subset_num_batches: 20
# seed: 17
# dist_timeout: 600.0

# models:
# -
#   model_name: ${{variables.model_name_or_path}}
#   model:
#     name: hf_causal_lm
#     pretrained_model_name_or_path: ${{variables.model_name_or_path}}
#     init_device: mixed
#     pretrained: true
#     device_map: auto
#     torch_dtype: float16
#   tokenizer:
#     name: ${{variables.model_name_or_path}}
#     kwargs:
#       model_max_length: ${{variables.max_seq_len}}
# """
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_yaml:
#             temp_yaml.write(peft_yaml_content)
#             eval_yaml = temp_yaml.name
#             logger.info(f"Created temporary YAML for PEFT model: {eval_yaml}")
#     else:
#         # Use standard YAML for non-PEFT models
#         eval_yaml = "eval/yamls/hf_eval.yaml"
    
#     # Use the exact same command structure as your team's Modal version
#     logger.info(f"\nEvaluating model at path: {model_dir}")
#     eval_cmd = [
#         "composer",
#         "eval/eval.py",
#         eval_yaml,
#         "icl_tasks=eval/yamls/copa.yaml",
#         f"results_path={save_path}"
#     ]
    
#     # Add model path only if not using PEFT YAML (where it's already included)
#     if not is_peft:
#         eval_cmd.append(f"variables.model_name_or_path={model_dir}")
    
#     logger.info(f"Running command: {' '.join(map(str, eval_cmd))}")
#     result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
#     logger.info(result.stdout)
#     if result.stderr:
#         logger.error(f"Evaluation errors: {result.stderr}")
    
#     # Clean up temp file if we created one
#     if is_peft and os.path.exists(eval_yaml):
#         os.unlink(eval_yaml)
        
#     logger.info("Evaluation complete!")

    
# def evaluate_model(checkpoint_path: str):
#     """Evaluate a model using Composer's eval script"""
#     import subprocess, os
    
#     # Get HF token for model access
#     get_hf_token()
    
#     # Get scripts directory
#     scripts_dir = os.path.join(ROOT_DIR, "scripts")
#     if not os.path.exists(scripts_dir):
#         logger.error(f"Scripts directory not found at {scripts_dir}")
#         return
    
#     # Construct path similar to Modal version
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
    
#     # Use the exact same command structure as your team's Modal version
#     logger.info(f"\nEvaluating model at path: {model_dir}")
#     eval_cmd = [
#         "composer",
#         "eval/eval.py",
#         "eval/yamls/hf_eval.yaml",
#         "icl_tasks=eval/yamls/copa.yaml",
#         f"variables.model_name_or_path={model_dir}",
#         f"results_path={save_path}"
#     ]
    
#     # If this is a PEFT model, add specific config overrides
#     if is_peft:
#         eval_cmd.extend([
#             # Set device mapping for adapter loading
#             "models.0.model.device_map=auto",
#             # Load in float16 to conserve memory
#             "models.0.model.torch_dtype=float16"   
#         ])
    
#     logger.info(f"Running command: {' '.join(map(str, eval_cmd))}")
#     result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
#     logger.info(result.stdout)
#     if result.stderr:
#         logger.error(f"Evaluation errors: {result.stderr}")
    
#     logger.info("Evaluation complete!")

# def evaluate_model(checkpoint_path: str):
#     """Evaluate a model using Composer's eval script"""
#     import subprocess, os, json
#     from pathlib import Path
    
#     # Get HF token for model access
#     get_hf_token()
    
#     # Get scripts directory
#     scripts_dir = os.path.join(ROOT_DIR, "scripts")
#     if not os.path.exists(scripts_dir):
#         logger.error(f"Scripts directory not found at {scripts_dir}")
#         return
    
#     # Construct path similar to Modal version
#     checkpoint_dir = os.path.join(ROOT_DIR, "model-checkpoints")
#     model_dir = os.path.join(checkpoint_dir, checkpoint_path)
#     save_path = os.path.join(model_dir, "evals")
    
#     # Ensure output directory exists
#     os.makedirs(save_path, exist_ok=True)
    
#     # # Check if this is a PEFT model
#     # is_peft = os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    
#     # # For PEFT models, we need a different approach
#     # if is_peft:
#     #     # Check if we should skip eval for PEFT models (they require special handling)
#     #     logger.info("Detected PEFT adapter model - using simplified evaluation")
#     #     return evaluate_adapter_simple(model_dir, save_path)
    
#     # Change to scripts directory
#     os.chdir(scripts_dir)
#     logger.info(f"Working directory: {os.getcwd()}")
    
#     # Use the exact same command structure as your team's Modal version
#     logger.info(f"\nEvaluating model at path: {model_dir}")
#     eval_cmd = [
#         "composer",
#         "eval/eval.py",
#         "eval/yamls/hf_eval.yaml",
#         "icl_tasks=eval/yamls/copa.yaml",
#         f"variables.model_name_or_path={model_dir}",  # No .absolute() method
#         f"results_path={save_path}",  # No .absolute() method and no tokenizer_name
#         f"variables.merge_lora={str(IS_PEFT).lower()}" # Responsible for adapter merging
#     ]
    
#     logger.info(f"Running command: {' '.join(map(str, eval_cmd))}")
#     result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
#     logger.info(result.stdout)
#     if result.stderr:
#         logger.error(f"Evaluation errors: {result.stderr}")
    
#     logger.info("Evaluation complete!")

def evaluate_adapter_simple(model_dir, save_path):
    """Simple evaluation for adapter models"""
    import tempfile
    import sys
    import subprocess
    
    # Create a simple script to test the adapter
    script = f"""
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import json
import os

# Create results directory
os.makedirs("{save_path}", exist_ok=True)

# Load model and tokenizer
print("Loading adapter model")
model = AutoPeftModelForCausalLM.from_pretrained(
    "{model_dir}",
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("{model_dir}")

# Simple prompts to test
test_prompts = [
    "The capital of France is",
    "To make chocolate chip cookies, you need"
]

results = []
for prompt in test_prompts:
    print(f"Testing prompt: {{prompt}}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    results.append({{"prompt": prompt, "completion": generated_text}})
    print(f"Output: {{generated_text}}")

# Save results to the evals directory
with open("{save_path}/simple_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Evaluation results saved to {save_path}/simple_eval_results.json")
"""
    
    # Save script to temporary file
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    
    try:
        # Run evaluation
        logger.info("Running simplified adapter evaluation")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.error(f"Evaluation errors: {result.stderr}")
    finally:
        # Clean up
        os.unlink(script_path)
    
    return "Adapter evaluation complete"
# def evaluate_model(checkpoint_path: str):
#     """Evaluate a model using Composer's eval script"""
#     import subprocess, os
#     from pathlib import Path
    
#     # Get HF token for model access
#     get_hf_token()
    
#     # Get scripts directory
#     scripts_dir = os.path.join(ROOT_DIR, "scripts")
#     if not os.path.exists(scripts_dir):
#         logger.error(f"Scripts directory not found at {scripts_dir}")
#         return
    
#     # Construct ABSOLUTE path to model directory
#     checkpoint_dir = os.path.join(ROOT_DIR, "model-checkpoints")
#     if "/" in str(checkpoint_path):
#         model_dir = Path(checkpoint_dir) / Path(checkpoint_path.split("/")[0])
#     else:
#         model_dir = Path(checkpoint_dir) / checkpoint_path
    
#     # Ensure model directory exists
#     if not model_dir.exists():
#         logger.error(f"Model directory {model_dir} does not exist")
#         return
    
#     # Check for tokenizer files and copy if needed
#     tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
#     has_tokenizer = all(os.path.exists(model_dir / file) for file in tokenizer_files)
    
#     if not has_tokenizer:
#         logger.info(f"Tokenizer files not found in {model_dir}, copying from base model")
#         try:
#             from transformers import AutoTokenizer
#             tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#             tokenizer.save_pretrained(model_dir)
#             logger.info(f"Tokenizer files saved to {model_dir}")
#         except Exception as e:
#             logger.error(f"Failed to save tokenizer: {e}")
#             return
    
#     # Create evals directory
#     save_path = model_dir / "evals"
#     save_path.mkdir(exist_ok=True)
    
#     # Change to scripts directory
#     os.chdir(scripts_dir)
#     logger.info(f"Working directory: {os.getcwd()}")
    
#     # Run evaluation with ABSOLUTE paths
#     logger.info(f"\nEvaluating model at absolute path: {model_dir.absolute()}")
#     eval_cmd = [
#         "composer",
#         "eval/eval.py",
#         "eval/yamls/hf_eval.yaml",
#         "icl_tasks=eval/yamls/copa.yaml",
#         f"variables.model_name_or_path={model_dir.absolute()}",  # Use variables namespace
#         f"variables.tokenizer_name={model_dir.absolute()}",      # Use variables namespace
#         f"results_path={save_path.absolute()}",
#     ]
    
#     logger.info(f"Running command: {' '.join(map(str, eval_cmd))}")
#     result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
#     logger.info(result.stdout)
#     if result.stderr:
#         logger.error(f"Evaluation errors: {result.stderr}")
    
#     logger.info("Evaluation complete!")



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
    import os
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
# def push_folder_to_hf(
#     folder_path: str, 
#     repo_id: Optional[str] = None, 
#     repo_type: str = "model", 
#     private: bool = True
# ) -> None:
#     """Upload model checkpoint to HuggingFace Hub."""
#     from huggingface_hub import HfApi
#     from pathlib import Path

#     folder_path = Path(folder_path)
#     if not folder_path.exists() or not folder_path.is_dir():
#         raise FileNotFoundError(
#             f"Folder {folder_path} does not exist or is not a directory."
#         )
#     folder_name = folder_path.name
#     if repo_id is None: 
#         repo_id = f"LocalResearchGroup/{folder_name}"

#     api = HfApi()

#     logger.info(f'Uploading {folder_path} to HuggingFace Hub at {repo_id}')
    
#     api.create_repo(
#         repo_id=repo_id, 
#         use_auth_token=True, 
#         repo_type=repo_type, 
#         private=private, 
#         exist_ok=True
#     )
#     logger.info('Repo created.')

#     api.upload_folder(
#         folder_path=folder_path, 
#         repo_id=repo_id, 
#         use_auth_token=True, 
#         repo_type=repo_type
#     )
#     logger.info(f'Folder "{folder_path}" uploaded to: "{repo_id}" successfully.')


def main():
    """Main entry point for the script"""
    from pathlib import Path
    import time
    
    # Create runs directory if it doesn't exist
    os.makedirs("./runs", exist_ok=True)
    
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting training run: {run_ts}")

    get_stats()
    time.sleep(1)
    cleanup_dataset()
    #convert_c4_small_dataset()  # Only run once

    model_path = train_model(run_ts, yaml_path=TRAIN_YAML)
    logger.info(f"Model path: {model_path}")
    model_path = Path(model_path).name
    time.sleep(1)
    
    view_model_checkpoints()
    time.sleep(1)

    convert_model_to_hf(model_path, upload_to_hf=False)
    time.sleep(1)
  
    evaluate_model(model_path)
    time.sleep(1)

    push_folder_to_hf(Path(model_path)) 
    time.sleep(1)

    generate_responses(model_path)
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main() 