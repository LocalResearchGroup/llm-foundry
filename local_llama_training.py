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


def extract_adapters(composer_checkpoint_path, output_path):
    """Convert a checkpoint using the built-in HF functionality, preserving adapter weights"""
    from composer.models.huggingface import write_huggingface_pretrained_from_composer_checkpoint
    import os
    from pathlib import Path
    
    try:
        # Create a temporary HF output path for the adapter extraction
        hf_output_path = str(Path(output_path).parent / f"{Path(output_path).name}-hf")
        os.makedirs(hf_output_path, exist_ok=True)
        
        # Use the built-in function that correctly extracts adapter weights
        write_huggingface_pretrained_from_composer_checkpoint(
            checkpoint_path=composer_checkpoint_path,
            output_folder=hf_output_path
        )
        
        # Copy adapter files to the original output path
        adapter_config = Path(hf_output_path) / "adapter_config.json"
        adapter_model = Path(hf_output_path) / "adapter_model.bin"
        
        if adapter_config.exists() and adapter_model.exists():
            import shutil
            shutil.copy(adapter_config, Path(output_path) / "adapter_config.json")
            shutil.copy(adapter_model, Path(output_path) / "adapter_model.bin")
            logger.info(f"Adapter files extracted and copied to {output_path}")
            return True
        else:
            logger.info("No adapter files found in HF output")
            return False
            
    except Exception as e:
        logger.info(f"Error extracting adapters: {e}")
        import traceback
        traceback.print_exc()
        return False

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
    if "/" in str(checkpoint_path) or "\\" in str(checkpoint_path):
        run_folder = checkpoint_path  # If full path provided
    else:
        run_folder = checkpoint_dir / checkpoint_path  # Just model name
    
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
    hf_output_path = run_folder#.parent / f"{run_folder.name}-hf"
    hf_output_path.mkdir(exist_ok=True, parents=True)

    # === USE THE BUILT-IN FUNCTION ===
    extract_adapters(composer_checkpoint_path=str(composer_checkpoint_path),
                     output_path=str(hf_output_path))

    path_tracker("AFTER_CONVERSION", check_paths=[
        hf_output_path,
        hf_output_path / "adapter_config.json",
        hf_output_path / "adapter_model.bin",
        run_folder / "adapter_config.json", 
        run_folder / "adapter_model.bin"
    ])
    
    # Add HF upload if requested
    if upload_to_hf:
        convert_cmd = [
            PYTHON_PATH, 
            str(os.path.join(scripts_dir, "inference", "convert_composer_to_hf.py")),
            "--composer_path", str(composer_checkpoint_path),
            "--hf_output_path", str(hf_output_path),
            "--hf_repo_for_upload", f"LocalResearchGroup/{run_folder.name}"
        ]
        
        logger.info(f"Uploading to HuggingFace: {' '.join(convert_cmd)}")
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.warning(f"Upload errors: {result.stderr}")
    
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
    from pathlib import Path
    
    # Get HF token for model access
    get_hf_token()
    
    # Get scripts directory
    scripts_dir = os.path.join(ROOT_DIR, "scripts")
    if not os.path.exists(scripts_dir):
        logger.error(f"Scripts directory not found at {scripts_dir}")
        return
    
    # Construct ABSOLUTE path to model directory
    checkpoint_dir = os.path.join(ROOT_DIR, "model-checkpoints")
    if "/" in str(checkpoint_path):
        model_dir = Path(checkpoint_dir) / Path(checkpoint_path.split("/")[0])
    else:
        model_dir = Path(checkpoint_dir) / checkpoint_path
    
    # Ensure model directory exists
    if not model_dir.exists():
        logger.error(f"Model directory {model_dir} does not exist")
        return
    
    # Check for tokenizer files and copy if needed
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    has_tokenizer = all(os.path.exists(model_dir / file) for file in tokenizer_files)
    
    if not has_tokenizer:
        logger.info(f"Tokenizer files not found in {model_dir}, copying from base model")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            tokenizer.save_pretrained(model_dir)
            logger.info(f"Tokenizer files saved to {model_dir}")
        except Exception as e:
            logger.error(f"Failed to save tokenizer: {e}")
            return
    
    # Create evals directory
    save_path = model_dir / "evals"
    save_path.mkdir(exist_ok=True)
    
    # Change to scripts directory
    os.chdir(scripts_dir)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Run evaluation with ABSOLUTE paths
    logger.info(f"\nEvaluating model at absolute path: {model_dir.absolute()}")
    eval_cmd = [
        "composer",
        "eval/eval.py",
        "eval/yamls/hf_eval.yaml",
        "icl_tasks=eval/yamls/copa.yaml",
        f"variables.model_name_or_path={model_dir.absolute()}",  # Use variables namespace
        f"variables.tokenizer_name={model_dir.absolute()}",      # Use variables namespace
        f"results_path={save_path.absolute()}",
    ]
    
    logger.info(f"Running command: {' '.join(map(str, eval_cmd))}")
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
    logger.info(result.stdout)
    if result.stderr:
        logger.error(f"Evaluation errors: {result.stderr}")
    
    logger.info("Evaluation complete!")

# def evaluate_model(checkpoint_path: str):
#     """Evaluate a model using Composer's eval script"""
#     import subprocess, os
#     from pathlib import Path

#     # Get scripts directory from ROOT_DIR
#     scripts_dir = os.path.join(ROOT_DIR, "scripts")
    
#     # Check if scripts directory exists
#     if not os.path.exists(scripts_dir):
#         logger.error(f"Scripts directory not found at {scripts_dir}")
#         return
        
#     # Change to scripts directory
#     os.chdir(scripts_dir)
#     logger.info(f"Working directory: {os.getcwd()}")
    
#     model_path = Path(f"../{checkpoint_path}")
#     save_path = model_path / "evals"  # Create evals subfolder path
    
#     logger.info("\nEvaluating model...")
#     eval_cmd = [
#         "composer",
#         "eval/eval.py",
#         "eval/yamls/hf_eval.yaml",
#         "icl_tasks=eval/yamls/copa.yaml",
#         f"variables.model_name_or_path={model_path}",
#         f"results_path={save_path}",  # Add results_path parameter
#     ]
#     result = subprocess.run(eval_cmd, capture_output=True, text=True)
#     logger.info(result.stdout)
#     if result.stderr:
#         logger.error(f"Evaluation errors: {result.stderr}")
    
#     os.chdir("..")  # Return to original directory
#     logger.info("Evaluation complete!")

# def evaluate_model(checkpoint_path: str) -> None:
#     """Evaluate the model on a benchmark"""
#     import subprocess
#     import os
#     from pathlib import Path
    
#     get_hf_token()
#     os.chdir("scripts")
#     logger.info(f"Working directory: {os.getcwd()}")
    
#     model_path = Path(f"../{checkpoint_path}")
#     save_path = model_path / "evals"  # Create evals subfolder path
    
#     logger.info("\nEvaluating model...")
#     eval_cmd = [
#         "composer",
#         "eval/eval.py",
#         "eval/yamls/hf_eval.yaml",
#         "icl_tasks=eval/yamls/copa.yaml",
#         f"variables.model_name_or_path={model_path}",
#         f"results_path={save_path}",  # Add results_path parameter
#     ]
#     result = subprocess.run(eval_cmd, capture_output=True, text=True)
#     logger.info(result.stdout)
#     if result.stderr:
#         logger.error(f"Evaluation errors: {result.stderr}")
    
#     os.chdir("..")  # Return to original directory
#     logger.info("Evaluation complete!")


def generate_responses(
    checkpoint_path: str, 
    prompts: Optional[Union[List[str], str]] = None
) -> None:
    """Generate responses from the model for given prompts"""
    import subprocess
    import os
    from pathlib import Path
    
    get_hf_token()
    os.chdir("scripts")
    logger.info(f"Working directory: {os.getcwd()}")
    
    model_path = Path(f"../{checkpoint_path}")

    if prompts is None:
        prompts = [
            "The answer to life, the universe, and happiness is",
            "Here's a quick recipe for baking chocolate chip cookies: Start by",
        ]
    elif isinstance(prompts, str):
        prompts = [prompts]

    logger.info("\nGenerating test responses...")
    generate_cmd = [
        PYTHON_PATH, "inference/hf_generate.py",
        "--name_or_path", model_path,
        "--max_new_tokens", "256",
        "--prompts",
        *prompts,
    ]
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.error(f"Generation errors: {result.stderr}")
    
    os.chdir("..")  # Return to original directory
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