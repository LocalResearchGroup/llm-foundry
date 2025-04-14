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
IS_PEFT = True
TRAIN_YAML="train/yamls/llama/llama3-1b-lora2.yaml"
OUTPUT_PRECISION ="bf16"
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



@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                       DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
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
    max_containers=1)
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



def download_model_if_needed(token, model_name_or_path):
    """Download the model if it's gated and requires a HuggingFace token"""
    import subprocess
    if token and "meta-llama" in model_name_or_path:
        print(f"Downloading model {model_name_or_path}...")
        local_model = "/tmp/llama-model"
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


def train_model(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml"):
    import os, subprocess, shutil
    from pathlib import Path
    
    # Change to llm-foundry/scripts directory at the start
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 2: Train the model

    print("\nTraining model...")
    model_name = get_model_name(yaml_path)
    run_folder = get_run_folder(run_ts, model_name)
    save_folder = Path(f"{run_folder}/native_checkpoints")

    save_folder.mkdir(exist_ok=True)
    shutil.copy(yaml_path, Path(save_folder) / Path(yaml_path).name)
    if USE_CUSTOM_MODEL:
        get_hf_token()
        download_model_if_needed(token=os.environ["HF_TOKEN"],model_name_or_path=model_name)
        os.environ["COMPOSER_SAVE_FOLDER"] = str(save_folder)
        print(f"Set COMPOSER_SAVE_FOLDER={save_folder}")
        
        # Run training with your adapter script
        train_cmd = [
            PYTHON_PATH,
            "train/train_with_llama_adapter.py",  # Use your adapter script
            yaml_path,
            f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small",
            f"save_folder={save_folder}",
            f"max_duration={TRAIN_DURATION}",
            f"save_interval={SAVE_INTERVAL}",
            "save_latest_filename=latest-rank0.pt",
            "model.should_save_peft_only=true",
        ]
        
        result = subprocess.run(train_cmd)
        print(f'Training complete for {run_ts}')
        print(f'Model checkpoints saved to {save_folder}')
    else:
        train_cmd = [
            "composer",
            "train/train.py",
            yaml_path,  # Updated YAML path
            f"save_folder={save_folder}",
        ]
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        print(result.stdout)
        print(f'Training complete for {run_ts}')
        print(f'Model checkpoints saved to {save_folder}')

    MODEL_CHECKPOINT_VOLUME.commit()

    # Print checkpoint file sizes
    view_model_checkpoints.remote(save_folder)
    
    if result.stderr:
        print("Training errors:", result.stderr)
    if result.returncode != 0:
        raise Exception(f"Training failed with exit code {result.returncode}\nStderr: {result.stderr}")
    return str(run_folder)

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


# @app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, secrets=[Secret.from_name("LRG")],
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
#                       DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
#               concurrency_limit=1)
@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                      DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def train_with_aim(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml"):
    import subprocess, time

    with modal.forward(43800) as tunnel:
        print(f"\nAim server available at: {tunnel.url}")
        model_path = None
        aim_task = run_aim_server(get_run_folder(run_ts, get_model_name(yaml_path)))
        time.sleep(5)
    
        try:
            model_path = train_model(run_ts, yaml_path)

        finally:
            aim_task.terminate()
            try:
                aim_task.wait(timeout=5)
            except subprocess.TimeoutExpired:
                aim_task.kill()
    
    return model_path


@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                       DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def view_model_checkpoints(checkpoint_dir=None):
    """View contents of model checkpoints directory"""
    import os
    from pathlib import Path
    
    if checkpoint_dir is None:
        checkpoint_dir = MODEL_CHECKPOINT_VOLUME_MOUNT_PATH
    
    checkpoint_dir = Path(checkpoint_dir)
    print(f"Viewing contents of {checkpoint_dir}")
    
    if checkpoint_dir.exists():
        # Find all files recursively
        for root, dirs, files in os.walk(checkpoint_dir):
            root_path = Path(root)
            print(f"\nDirectory: {root_path}")
            
            for file in files:
                file_path = root_path / file
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")
    else:
        print(f"Directory {checkpoint_dir} doesn't exist")
    
    return "Checkpoint viewing complete"

@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                       DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def convert_model_to_hf(checkpoint_path: str, upload_to_hf: bool = False):
    """Convert a model checkpoint to a HuggingFace format."""
    import subprocess, os
    from pathlib import Path

    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")

    run_folder = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path.split("/")[0]
    composer_checkpoint_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
    if composer_checkpoint_path.is_dir():
        composer_checkpoint_path = composer_checkpoint_path / "native_checkpoints" / "latest-rank0.pt"
    hf_output_path = run_folder

    print("\nConverting model to HuggingFace format...")
    convert_cmd = [
        PYTHON_PATH, "inference/convert_composer_to_hf.py",
        "--composer_path", composer_checkpoint_path,
        "--hf_output_path", hf_output_path,
        "--output_precision", f"{OUTPUT_PRECISION}",
        "--is_peft", f"{IS_PEFT}",
        "--train_yaml", f"{TRAIN_YAML}"
    ]
    if upload_to_hf: convert_cmd.extend(["--hf_repo_for_upload", f"LocalResearchGroup/{run_folder.name}"])

    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Conversion errors:", result.stderr)
    MODEL_CHECKPOINT_VOLUME.commit()
    print("Conversion complete!")

@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                       DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def cleanup_dataset():
    """Clean up corrupted dataset and create a fresh one."""
    import os, subprocess, shutil
    from pathlib import Path
    
    # Change to llm-foundry/scripts directory
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    # Check current dataset state
    data_path = Path(f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small")
    print(f"Examining dataset at {data_path}")
    
    if data_path.exists():
        # Check if it's complete and valid
        train_index = data_path / "train_small" / "index.json"
        val_index = data_path / "val_small" / "index.json"
        
        if train_index.exists() and val_index.exists():
            print("✅ Dataset appears to be complete and valid, no cleanup needed")
            return str(data_path)
        else:
            print("❌ Dataset is incomplete or corrupted, will remove and recreate")
            
            # Backup the old data just in case
            print("Making backup of existing data...")
            backup_dir = Path(f"{DATASETS_VOLUME_MOUNT_PATH}/c4_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
                    print(f"Warning during backup: {e}")
            
            # Remove the corrupted dataset
            try:
                shutil.rmtree(data_path)
                print(f"Removed corrupted dataset at {data_path}")
            except Exception as e:
                print(f"Error removing dataset: {e}")
                # If we can't remove, rename it
                try:
                    old_path = Path(f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small_corrupted")
                    shutil.move(data_path, old_path)
                    print(f"Renamed corrupted dataset to {old_path}")
                except Exception as e2:
                    print(f"Error renaming dataset: {e2}")
                    return "Failed to clean up dataset"
                
@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                       DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def evaluate_model(checkpoint_path: str):
    import subprocess, os
    from pathlib import Path
    get_hf_token()
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
    save_path = model_path/"evals"  # Create evals subfolder path
    
    print("\nEvaluating model...")
    eval_cmd = [
        "composer",
        "eval/eval.py",
        "eval/yamls/hf_eval.yaml",
        "icl_tasks=eval/yamls/copa.yaml",
        f"variables.model_name_or_path={model_path}",
        f"results_path={save_path}",  # Add results_path parameter
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Evaluation errors:", result.stderr)
    
    MODEL_CHECKPOINT_VOLUME.commit()  # Commit the new eval results
    print("Evaluation complete!")


@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                       DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def generate_responses(checkpoint_path: str, prompts: list[str]|str|None=None):
    import subprocess, os
    from pathlib import Path
    get_hf_token()
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
    

    print("\nGenerating test responses...")
    generate_cmd = [
        PYTHON_PATH, "inference/hf_generate.py",
        "--name_or_path", model_path,
        "--max_new_tokens", "256",
        "--prompts",
        *prompts,
    ]
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Generation errors:", result.stderr)
    print("Generation complete!")

@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                       DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def push_folder_to_hf(folder_path: str, repo_id: str | None = None, repo_type: str = "model", private: bool = True):
    """Upload model checkpoint to HuggingFace Hub."""
    from huggingface_hub import HfApi
    from pathlib import Path

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

    get_stats.remote()
    time.sleep(1)
    cleanup_dataset.remote()
    #convert_c4_small_dataset.remote() # Only run once

    model_path = train_with_aim.remote(run_ts, yaml_path="train/yamls/llama/llama3-1b-lora2.yaml")
    print(f"Model path: {model_path}")
    model_path = Path(model_path).name
    time.sleep(1)
    
    view_model_checkpoints.remote()
    time.sleep(1)

    convert_model_to_hf.remote(model_path, upload_to_hf=False)
    time.sleep(1)
  
    evaluate_model.remote(model_path)
    # time.sleep(1)

    push_folder_to_hf.remote(Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/model_path) 
    # time.sleep(1)

    generate_responses.remote(model_path)







# def train_with_hf_token(run_ts: str, model_name_or_path: str = "meta-llama/Llama-3.2-1B"):
#     """Train a model with LoRA adapters using HF token for access"""
#     import os, subprocess, yaml
#     from pathlib import Path
#     import sys
    
#     # Change to llm-foundry/scripts directory
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     # Get HF token and inject it into environment
#     token = get_hf_token()

#     # 1. Create custom_llama_model.py with our implementation
#     print("Creating custom_llama_model.py...")
#     custom_model_code = """
# import torch
# from composer.models import HuggingFaceModel
# from pathlib import Path
# import os
# import json

# # Import your custom model implementation
# from transformers import AutoModelForCausalLM

# def create_llama_composer_model(
#     pretrained_model_name_or_path,
#     tokenizer,
#     peft_config=None,
#     use_flash_attention_2=False,
#     trust_remote_code=True,
#     **kwargs
# ):
#     \"\"\"Create a Composer-compatible LlamaForCausalLM model with adapter support.\"\"\"
#     print(f"Creating LlamaForCausalLM from {pretrained_model_name_or_path}")
    
#     # Create the base model
#     model_kwargs = {
#         "trust_remote_code": trust_remote_code,
#     }
    
#     if use_flash_attention_2:
#         model_kwargs["attn_implementation"] = "flash_attention_2"
    
#     # Instantiate model from HF
#     llama_model = AutoModelForCausalLM.from_pretrained(
#         pretrained_model_name_or_path,
#         **model_kwargs
#     )
    
#     print(f"Created model: {type(llama_model).__name__}")
    
#     # Wrap it with Composer's HuggingFaceModel
#     composer_model = HuggingFaceModel(
#         model=llama_model,
#         tokenizer=tokenizer,
#         use_logits=False,
#         shift_labels=True,  # For causal LM
#         peft_config=peft_config,
#         should_save_peft_only=True,  # Save only adapter weights
#         **kwargs
#     )
    
#     # Add an on_save_checkpoint hook for adapter extraction
#     original_on_save = getattr(composer_model, 'on_save_checkpoint', None)
    
#     def on_save_checkpoint(self, checkpoint_path):
#         \"\"\"Extract adapter weights to standard format during checkpointing\"\"\"
#         print(f"Saving checkpoint to {checkpoint_path}")
        
#         # Call original method if it exists
#         if original_on_save:
#             original_on_save(checkpoint_path)
        
#         # Only extract adapters if using PEFT
#         if not hasattr(self.model, 'peft_config'):
#             print("Not using PEFT, skipping adapter extraction")
#             return
            
#         try:
#             from peft import get_peft_model_state_dict
            
#             # Extract adapter weights
#             adapter_state_dict = get_peft_model_state_dict(self.model)
            
#             # Create adapter files in parent directory
#             checkpoint_dir = Path(checkpoint_path).parent
#             output_dir = checkpoint_dir.parent
            
#             config_path = output_dir / "adapter_config.json"
#             weights_path = output_dir / "adapter_model.bin"
            
#             # Create adapter config
#             config_dict = {
#                 "base_model_name_or_path": pretrained_model_name_or_path,
#                 "peft_type": "LORA",
#                 "task_type": "CAUSAL_LM",
#                 "r": peft_config.get("r", 8),
#                 "target_modules": peft_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
#                 "lora_alpha": peft_config.get("lora_alpha", 16),
#                 "lora_dropout": peft_config.get("lora_dropout", 0.05),
#                 "inference_mode": False
#             }
            
#             with open(config_path, "w") as f:
#                 json.dump(config_dict, f, indent=2)
                
#             torch.save(adapter_state_dict, weights_path)
            
#             print(f"Saved adapter files to {output_dir}")
#             print(f"  - Config: {config_path}")
#             print(f"  - Weights: {weights_path}")
            
#         except Exception as e:
#             print(f"Error extracting adapter: {e}")
#             import traceback
#             traceback.print_exc()
    
#     # Attach the method to the instance
#     composer_model.on_save_checkpoint = on_save_checkpoint.__get__(composer_model)
    
#     return composer_model
# """
    
#     with open("/llm-foundry/scripts/custom_llama_model.py", "w") as f:
#         f.write(custom_model_code)
    
#     # 2. Download model if using a gated model
#     if token and "meta-llama" in model_name_or_path:
#         print(f"Downloading model {model_name_or_path}...")
#         local_model = "/tmp/llama-model"
#         download_cmd = [
#             PYTHON_PATH, "-c",
#             f"""
# import os
# from huggingface_hub import snapshot_download, login
# token = "{token}"
# login(token=token)
# local_dir = "{local_model}"
# print(f"Downloading model to {{local_dir}}")
# snapshot_download(repo_id="{model_name_or_path}", local_dir=local_dir, token=token)
# print("Download complete!")
#             """
#         ]
#         subprocess.run(download_cmd, check=True)
        
#         # Use local model path
#         model_name_or_path = local_model
    
#     # 3. Create a training script using custom_llama_model
#     print("Creating training script...")
#     training_script = f"""
# import os
# import sys
# from transformers import AutoTokenizer
# from pathlib import Path
# import torch

# # Add parent directory to path
# sys.path.append('/llm-foundry/scripts')

# # Import our custom model creator
# from custom_llama_model import create_llama_composer_model

# # Import composer stuff
# from composer import Trainer
# from composer.callbacks import LRMonitor, JSONLogger
# from composer.loggers import WandBLogger, FileLogger
# from composer.optim import DecoupledAdamW
# from composer.algorithms import GradientClipping
# from composer.utils import dist, reproducibility
# from composer.utils.misc import is_model_fsdp

# # Import data stuff
# from llmfoundry.data.text_data import build_text_dataloader

# def main():
#     # Set seed for reproducibility
#     reproducibility.seed_all(42)
    
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("{model_name_or_path}")
    
#     # Configure PEFT (LoRA)
#     peft_config = {{
#         "r": 8,
#         "peft_type": "LORA",
#         "task_type": "CAUSAL_LM",
#         "lora_alpha": 16, 
#         "lora_dropout": 0.05,
#         "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
#     }}
    
#     # Create our custom model
#     model = create_llama_composer_model(
#         pretrained_model_name_or_path="{model_name_or_path}",
#         tokenizer=tokenizer,
#         peft_config=peft_config,
#         use_flash_attention_2=True,
#         allow_embedding_resizing=True
#     )
    
#     # Create train dataloader
#     train_loader = build_text_dataloader(
#         dataset_name="text",
#         tokenizer=tokenizer,
#         dataset_config={{
#             "local": "/datasets/c4_small",
#             "split": "train_small"
#         }},
#         drop_last=True,
#         shuffle=True,
#         max_seq_len=2048,
#         global_batch_size=2
#     )
    
#     # Create eval dataloader
#     eval_loader = build_text_dataloader(
#         dataset_name="text",
#         tokenizer=tokenizer,
#         dataset_config={{
#             "local": "/datasets/c4_small",
#             "split": "val_small"
#         }},
#         drop_last=False,
#         shuffle=False,
#         max_seq_len=2048,
#         global_batch_size=2
#     )
    
#     # Create optimizer
#     optimizer = DecoupledAdamW(
#         model.parameters(),
#         lr=2e-5,
#         betas=(0.9, 0.999),
#         eps=1e-8,
#         weight_decay=0.01
#     )
    
#     # Create algorithms
#     algorithms = [
#         GradientClipping(clipping_type="norm", clipping_threshold=1.0)
#     ]
    
#     # Create loggers
#     loggers = [
#         FileLogger(log_dir="{run_folder}", log_interval=1)
#     ]
    
#     # Instantiate trainer
#     trainer = Trainer(
#         model=model,
#         train_dataloader=train_loader,
#         eval_dataloader=eval_loader,
#         max_duration="2ba",  # 2 batches for testing
#         optimizers=optimizer,
#         algorithms=algorithms,
#         loggers=loggers,
#         save_folder="{save_folder}",
#         save_interval="1ba",
#         save_latest_filename="latest-rank0.pt",
#         save_overwrite=True,
#         device="gpu" if torch.cuda.is_available() else "cpu",
#         eval_interval="1ba"
#     )
    
#     # Start training
#     print("Starting training...")
#     trainer.fit()
#     print("Training complete!")

# if __name__ == "__main__":
#     main()
# """
    
#     # Create directories
#     model_name = "llama-lora"
#     run_folder = get_run_folder(run_ts, model_name)
#     save_folder = Path(f"{run_folder}/native_checkpoints")
#     save_folder.mkdir(exist_ok=True, parents=True)
    
#     # Write training script
#     script_path = "/llm-foundry/scripts/train_custom.py"
#     with open(script_path, "w") as f:
#         f.write(training_script.replace("{run_folder}", run_folder).replace("{save_folder}", str(save_folder)))
    
#     # 4. Run the training script
#     print("\nRunning training script...")
#     env = os.environ.copy()
#     if token:
#         env["HUGGINGFACE_TOKEN"] = token
#         env["HF_TOKEN"] = token
#         env["HUGGINGFACE_HUB_TOKEN"] = token
    
#     train_cmd = [PYTHON_PATH, script_path]
#     result = subprocess.run(train_cmd, env=env)
    
#     if result.returncode != 0:
#         print(f"Training failed with exit code {result.returncode}")
#     else:
#         print("Training completed successfully")
    
#     # 5. Check for adapter files
#     print("\nChecking for adapter files...")
#     adapter_config = Path(run_folder) / "adapter_config.json"
#     adapter_weights = Path(run_folder) / "adapter_model.bin"
    
#     if adapter_config.exists() and adapter_weights.exists():
#         print(f"✅ Adapter files found:")
#         print(f"  - Config: {adapter_config}")
#         print(f"  - Weights: {adapter_weights} ({os.path.getsize(adapter_weights)/1024/1024:.2f} MB)")
#     else:
#         print("❌ Adapter files not found, checking for checkpoints...")
        
#         # Try to extract adapters from checkpoint
#         checkpoint_path = save_folder / "latest-rank0.pt"
#         if checkpoint_path.exists():
#             print(f"Found checkpoint at {checkpoint_path}, extracting adapters...")
#             extract_cmd = [
#                 PYTHON_PATH, "-c",
#                 f"""
# import torch
# import json
# from pathlib import Path
# from peft import get_peft_model_state_dict

# checkpoint = torch.load("{checkpoint_path}", map_location="cpu")
# if "state" in checkpoint and "model" in checkpoint["state"]:
#     model_state = checkpoint["state"]["model"]
#     lora_state = {{k: v for k, v in model_state.items() if "lora_" in k}}
    
#     if lora_state:
#         print(f"Found {{len(lora_state)}} LoRA weights")
        
#         # Create adapter files
#         config_path = Path("{run_folder}") / "adapter_config.json"
#         weights_path = Path("{run_folder}") / "adapter_model.bin"
        
#         # Create adapter config
#         config_dict = {{
#             "base_model_name_or_path": "{model_name_or_path}",
#             "peft_type": "LORA",
#             "task_type": "CAUSAL_LM",
#             "r": 8,
#             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
#             "lora_alpha": 16,
#             "lora_dropout": 0.05,
#             "inference_mode": False
#         }}
        
#         with open(config_path, "w") as f:
#             json.dump(config_dict, f, indent=2)
            
#         torch.save(lora_state, weights_path)
        
#         print(f"Created adapter files at {{config_path}} and {{weights_path}}")
#     else:
#         print("No LoRA weights found in checkpoint")
# else:
#     print("Checkpoint doesn't have expected structure")
#                 """
#             ]
#             subprocess.run(extract_cmd)
    
#     # Commit volume changes
#     MODEL_CHECKPOINT_VOLUME.commit()
#     return str(run_folder)