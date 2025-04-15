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

# Local paths
DATASET_BASE_PATH = "./datasets"  # Local dataset path
MODEL_CHECKPOINT_PATH = "./model-checkpoints"  # Local model checkpoint path
IS_PEFT = True
# Update the path to match your actual directory structure
TRAIN_YAML = "yamls/llama/llama3-1b-lora2.yaml"  # Adjusted path
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


def check_llmfoundry_installed():
    """Check if llmfoundry is installed and provide instructions if not"""
    try:
        import llmfoundry
        logger.info(f"llmfoundry version {llmfoundry.__version__} is installed")
        return True
    except ImportError:
        logger.error("llmfoundry is not installed or not in the Python path")
        logger.error("Please install it in development mode with:")
        logger.error("  pip install -e .")
        logger.error("Run this command from the root directory of the llm-foundry repository")
        return False


def get_model_name(yaml_path: str) -> str:
    """Extract model name from YAML path"""
    return Path(yaml_path).stem


def get_run_folder(run_ts: str, model_name: str) -> str:
    """Get folder path for run artifacts"""
    return f"./runs/{model_name}-{run_ts}"


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
            [PYTHON_PATH, "-c", "import flash_attn; print(flash_attn.__version__)"],
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
        local_model = "./models/llama-model"
        os.makedirs(local_model, exist_ok=True)
        
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
        subprocess.run(download_cmd, check=True)
        return local_model
    return model_name_or_path


def train_model(run_ts: str, yaml_path: str = "yamls/llama/llama3-1b-lora2.yaml") -> str:
    """Train the model using the specified YAML configuration"""
    import os
    import subprocess
    import shutil
    import sys
    from pathlib import Path
    
    # Add the parent directory to Python path so we can find llmfoundry
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        logger.info(f"Added {root_dir} to Python path")
    
    # Get absolute paths
    scripts_dir = os.path.join(root_dir, "scripts")
    yaml_path = os.path.join(scripts_dir, yaml_path)
    
    # Change to llm-foundry/scripts directory at the start
    os.chdir(scripts_dir)
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Step 2: Train the model
    logger.info("\nTraining model...")
    model_name = get_model_name(yaml_path)
    run_folder = get_run_folder(run_ts, model_name)
    save_folder = Path(f"../{run_folder}/native_checkpoints")

    save_folder.mkdir(exist_ok=True, parents=True)
    shutil.copy(yaml_path, Path(save_folder) / Path(yaml_path).name)
    if USE_CUSTOM_MODEL:
        logger.info("Looking for HuggingFace token...")
        get_hf_token()
        download_model_if_needed(token=os.environ["HF_TOKEN"], model_name_or_path=model_name)
        os.environ["COMPOSER_SAVE_FOLDER"] = str(save_folder)
        logger.info(f"Set COMPOSER_SAVE_FOLDER={save_folder}")
        
        # Run training with our custom script
        train_cmd = [
            PYTHON_PATH,
            "train/train_with_custom_llama.py",  # Use our new custom script
            yaml_path,
            "../datasets/c4_small",  # Use path relative to scripts dir
            f"save_folder={save_folder}",
            f"max_duration={TRAIN_DURATION}",
            f"save_interval={SAVE_INTERVAL}",
            "save_latest_filename=latest-rank0.pt",
            "model.should_save_peft_only=true",
            "max_seq_len=2048",  # Add explicit max_seq_len parameter at root level
        ]
        
        result = subprocess.run(train_cmd)
        logger.info(f'Training complete for {run_ts}')
        logger.info(f'Model checkpoints saved to {save_folder}')
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


def convert_model_to_hf(checkpoint_path: str, upload_to_hf: bool = False) -> None:
    """Convert a model checkpoint to a HuggingFace format."""
    import subprocess
    import os
    from pathlib import Path

    os.chdir("scripts")
    logger.info(f"Working directory: {os.getcwd()}")

    run_folder = Path(f"../{checkpoint_path.split('/')[0]}")
    composer_checkpoint_path = Path(f"../{checkpoint_path}")
    if composer_checkpoint_path.is_dir():
        composer_checkpoint_path = (
            composer_checkpoint_path / "native_checkpoints" / "latest-rank0.pt"
        )
    hf_output_path = run_folder

    logger.info("\nConverting model to HuggingFace format...")
    convert_cmd = [
        PYTHON_PATH, "inference/convert_composer_to_hf.py",
        "--composer_path", composer_checkpoint_path,
        "--hf_output_path", hf_output_path,
        "--output_precision", f"{OUTPUT_PRECISION}",
        "--is_peft", f"{IS_PEFT}",
        "--train_yaml", f"{TRAIN_YAML}"
    ]
    if upload_to_hf: 
        convert_cmd.extend([
            "--hf_repo_for_upload", 
            f"LocalResearchGroup/{run_folder.name}"
        ])

    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.error(f"Conversion errors: {result.stderr}")
    
    os.chdir("..")  # Return to original directory
    logger.info("Conversion complete!")


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


def evaluate_model(checkpoint_path: str) -> None:
    """Evaluate the model on a benchmark"""
    import subprocess
    import os
    from pathlib import Path
    
    get_hf_token()
    os.chdir("scripts")
    logger.info(f"Working directory: {os.getcwd()}")
    
    model_path = Path(f"../{checkpoint_path}")
    save_path = model_path / "evals"  # Create evals subfolder path
    
    logger.info("\nEvaluating model...")
    eval_cmd = [
        "composer",
        "eval/eval.py",
        "eval/yamls/hf_eval.yaml",
        "icl_tasks=eval/yamls/copa.yaml",
        f"variables.model_name_or_path={model_path}",
        f"results_path={save_path}",  # Add results_path parameter
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.error(f"Evaluation errors: {result.stderr}")
    
    os.chdir("..")  # Return to original directory
    logger.info("Evaluation complete!")


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


def push_folder_to_hf(
    folder_path: str, 
    repo_id: Optional[str] = None, 
    repo_type: str = "model", 
    private: bool = True
) -> None:
    """Upload model checkpoint to HuggingFace Hub."""
    from huggingface_hub import HfApi
    from pathlib import Path

    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(
            f"Folder {folder_path} does not exist or is not a directory."
        )
    folder_name = folder_path.name
    if repo_id is None: 
        repo_id = f"LocalResearchGroup/{folder_name}"

    api = HfApi()

    logger.info(f'Uploading {folder_path} to HuggingFace Hub at {repo_id}')
    
    api.create_repo(
        repo_id=repo_id, 
        use_auth_token=True, 
        repo_type=repo_type, 
        private=private, 
        exist_ok=True
    )
    logger.info('Repo created.')

    api.upload_folder(
        folder_path=folder_path, 
        repo_id=repo_id, 
        use_auth_token=True, 
        repo_type=repo_type
    )
    logger.info(f'Folder "{folder_path}" uploaded to: "{repo_id}" successfully.')


def main():
    """Main entry point for the script"""
    from pathlib import Path
    import time
    
    # Create runs directory if it doesn't exist
    os.makedirs("./runs", exist_ok=True)
    
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting training run: {run_ts}")

    # Check if llmfoundry is installed
    if not check_llmfoundry_installed():
        logger.error("Cannot proceed without llmfoundry installed")
        return

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