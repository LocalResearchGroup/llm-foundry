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


########################33
@app.function(image=image)
def debug_adapter_code():
    """Examine the composer_llama_adapter.py file and add diagnostics"""
    import os, shutil
    
    adapter_file = "/llm-foundry/llmfoundry/models/llama/composer_llama_adapter.py"
    
    # First create a backup
    backup_file = adapter_file + ".backup"
    shutil.copy(adapter_file, backup_file)
    print(f"Created backup at {backup_file}")
    
    # Read the file
    with open(adapter_file, "r") as f:
        content = f.read()
    
    # Check key elements
    print("Checking adapter code:")
    print(f"- Contains get_state_dict: {'get_state_dict' in content}")
    print(f"- Contains get_peft_model_state_dict: {'get_peft_model_state_dict' in content}")
    print(f"- Contains on_save_checkpoint: {'on_save_checkpoint' in content}")
    
    # Add extensive debug logging
    modified_content = content.replace(
        "def get_state_dict(self, state_dict=None):",
        """def get_state_dict(self, state_dict=None):
        print("\\n[DEBUG] get_state_dict called!")
        try:"""
    ).replace(
        "return regular_state_dict",
        """            return regular_state_dict
        except Exception as e:
            print(f"[DEBUG] Error in get_state_dict: {e}")
            import traceback
            traceback.print_exc()
            return regular_state_dict"""
    )
    
    # Check if on_save_checkpoint exists, if not add it
    if "on_save_checkpoint" not in content:
        # Find the end of the class definition
        class_end = modified_content.rfind("def get_state_dict")
        class_end = modified_content.rfind("}", class_end) if class_end > 0 else len(modified_content)
        
        # Add on_save_checkpoint method
        on_save_method = """
def on_save_checkpoint(self, checkpoint_path):
    "Hook explicitly called when saving checkpoints"
    print(f"\\n[DEBUG] on_save_checkpoint called for path: {checkpoint_path}")
    
    if not hasattr(self, 'using_peft') or not self.using_peft:
        print("[DEBUG] Not a PEFT model, skipping adapter saving")
        return
    
    print("[DEBUG] Extracting adapter weights...")
    adapter_state_dict = {}
    
    try:
        # Try to extract adapter weights
        if hasattr(self.model, 'get_adapter_state_dict'):
            print("[DEBUG] Using model.get_adapter_state_dict()")
            adapter_state_dict = self.model.get_adapter_state_dict()
        else:
            print("[DEBUG] Using get_peft_model_state_dict()")
            from peft import get_peft_model_state_dict
            adapter_state_dict = get_peft_model_state_dict(self.model)
        
        # Check if we have adapter weights
        if not adapter_state_dict:
            print("[DEBUG] No adapter weights found!")
            print("[DEBUG] Model attributes:", dir(self.model))
            print("[DEBUG] Is PEFT model:", hasattr(self.model, 'peft_config'))
            return
        
        print(f"[DEBUG] Found {len(adapter_state_dict)} adapter weights")
        
        # Save adapter files
        import os, json
        from pathlib import Path
        
        # Get parent directory from checkpoint path
        parent_folder = Path(os.environ.get("COMPOSER_SAVE_FOLDER", ""))
        if not parent_folder.exists():
            parent_folder = Path(checkpoint_path).parent.parent
        
        print(f"[DEBUG] Saving adapter files to {parent_folder}")
        
        # Create adapter config
        config_dict = {
            "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 8,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "inference_mode": False
        }
        
        # Save files
        import torch
        config_path = parent_folder / "adapter_config.json"
        weights_path = parent_folder / "adapter_model.bin"
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"[DEBUG] Saved adapter config to {config_path}")
        
        torch.save(adapter_state_dict, weights_path)
        print(f"[DEBUG] Saved adapter weights to {weights_path}")
        
        # Also try saving to checkpoint directory itself
        checkpoint_dir = Path(checkpoint_path).parent
        config_path2 = checkpoint_dir / "adapter_config.json"
        weights_path2 = checkpoint_dir / "adapter_model.bin"
        
        with open(config_path2, "w") as f:
            json.dump(config_dict, f, indent=2)
        torch.save(adapter_state_dict, weights_path2)
        print(f"[DEBUG] Also saved adapter files to {checkpoint_dir}")
        
    except Exception as e:
        print(f"[DEBUG] Error in on_save_checkpoint: {e}")
        import traceback
        traceback.print_exc()
"""
        # Insert on_save_method at the correct position
        modified_content = modified_content[:class_end] + on_save_method + modified_content[class_end:]
    
    # Add extra code to verify PEFT setup in initialization method
    modified_content = modified_content.replace(
        "print(\"Initializing CustomLlamaModel\")",
        """print("Initializing CustomLlamaModel")
                # Debug PEFT configuration
                print(f"PEFT config: {peft_config}")"""
    ).replace(
        "print(\"CustomLlamaModel initialization complete\")",
        """print("CustomLlamaModel initialization complete")
                # Verify PEFT was applied
                if peft_config is not None:
                    print("Checking if PEFT was successfully applied:")
                    print(f"- Model has peft_config attribute: {hasattr(self.model, 'peft_config')}")
                    if hasattr(self.model, 'peft_config'):
                        print(f"- PEFT config type: {type(self.model.peft_config)}")
                    else:
                        print("WARNING: PEFT may not have been applied correctly!")"""
    )
    
    # Write modified file
    with open(adapter_file, "w") as f:
        f.write(modified_content)
    
    print("\nAdded extensive debugging to adapter file")
    
    # Also check train_with_llama_adapter.py to ensure it's importing properly
    adapter_import_file = "/llm-foundry/scripts/train/train_with_llama_adapter.py"
    with open(adapter_import_file, "r") as f:
        import_content = f.read()
    print("\nChecking train_with_llama_adapter.py:")
    print(f"- Imports our adapter: {'composer_llama_adapter' in import_content}")
    
    return "Adapter code instrumented with debugging"

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
             secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
             max_containers=1)
def check_token():
    """Verify HF token exists and works."""
    import os
    import subprocess
    from huggingface_hub import HfApi, whoami
    
    # Get token using your existing function
    token = get_hf_token()
    
    print(f"Token retrieved: {'Yes' if token else 'No'}")
    if not token:
        print("⚠️ No token found!")
        return False
        
    # Test token with a simple API call
    try:
        api = HfApi(token=token)
        user_info = whoami(token=token)
        print(f"✅ Token verified! Logged in as: {user_info['name']}")
        
        # Test access to Llama model specifically
        print("Testing access to Llama model...")
        cmd = [
            PYTHON_PATH, "-c", 
            f"from huggingface_hub import hf_hub_download; "
            f"file = hf_hub_download('meta-llama/Llama-3.2-1B', 'config.json', token='{token}', local_dir='/tmp/test', force_download=True); "
            f"print('Success! Downloaded:', file)"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully accessed Llama model")
            print(result.stdout)
            return token
        else:
            print("❌ Failed to access Llama model")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Token verification failed: {e}")
        return False

@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                      DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def train_with_hf_token(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml"):
    """Train using verified HF token"""
    import os, subprocess, yaml, shutil
    from pathlib import Path
    
    # Change to llm-foundry directory
    os.chdir("/llm-foundry/scripts")
    
    # Get token and download model
    token = get_hf_token()
    if token:
        # Download model files first
        local_model = "/tmp/llama-3-2-1b"
        download_cmd = [
            PYTHON_PATH, "-c",
            f"""
import os
from huggingface_hub import snapshot_download, login
token = "{token}"
login(token=token)
local_dir = "{local_model}"
print(f"Downloading model to {{local_dir}}")
snapshot_download(repo_id="meta-llama/Llama-3.2-1B", local_dir=local_dir, token=token)
print("Download complete!")
            """
        ]
        subprocess.run(download_cmd, check=True)
    
    # Load existing YAML and add missing fields
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    # Add missing required fields
    config['global_train_batch_size'] = 2
    config['device_train_microbatch_size'] = 1
    config['device_eval_batch_size'] = 1
    
    # Ensure train_loader and eval_loader are properly configured
    if 'train_loader' not in config:
        config['train_loader'] = {
            'name': 'text',
            'dataset': {
                'local': "${variables.data_local}",
                'split': 'train_small'
            }
        }
    elif 'name' not in config['train_loader']:
        config['train_loader']['name'] = 'text'
    
    if 'eval_loader' not in config:
        config['eval_loader'] = {
            'name': 'text',
            'dataset': {
                'local': "${variables.data_local}",
                'split': 'val_small'
            }
        }
    elif 'name' not in config['eval_loader']:
        config['eval_loader']['name'] = 'text'
    
    # If token exists, modify config to use local model
    if token:
        config['model']['pretrained_model_name_or_path'] = local_model
    
    # Set up paths
    model_name = get_model_name(yaml_path)
    run_folder = get_run_folder(run_ts, model_name)
    save_folder = Path(f"{run_folder}/native_checkpoints")
    save_folder.mkdir(exist_ok=True, parents=True)
    
    # Save modified YAML
    fixed_yaml_path = f"{save_folder}/fixed_config.yaml"
    with open(fixed_yaml_path, "w") as f:
        yaml.dump(config, f)
    
    # Set environment variable for adapter saving
    os.environ["COMPOSER_SAVE_FOLDER"] = str(save_folder)
    print(f"Set COMPOSER_SAVE_FOLDER={save_folder}")
    
    # Run training with modified YAML
    train_cmd = [
        PYTHON_PATH,
        "train/train_with_llama_adapter.py",
        fixed_yaml_path,
        f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small",
        f"save_folder={save_folder}",
        f"max_duration={TRAIN_DURATION}",
        f"save_interval={SAVE_INTERVAL}",
        "save_latest_filename=latest-rank0.pt",
        "model.should_save_peft_only=true",
    ]
    
    # Set token in environment
    env = os.environ.copy()
    if token:
        env["HUGGINGFACE_TOKEN"] = token
        env["HF_TOKEN"] = token
        env["HUGGINGFACE_HUB_TOKEN"] = token
    
    subprocess.run(train_cmd, env=env, check=True)
       # Check for adapter files in multiple places
    print("\nSearching for adapter files:")
    locations_to_check = [
        Path(run_folder),
        Path(save_folder),
        Path(save_folder) / "checkpoints"
    ]
    
    adapter_files_found = False
    for location in locations_to_check:
        if not location.exists():
            print(f"Location does not exist: {location}")
            continue
            
        print(f"\nChecking {location}:")
        files = os.listdir(location)
        print(f"Files: {files}")
        
        adapter_files = [f for f in files if 'adapter' in f]
        if adapter_files:
            print(f"✅ Found adapter files: {adapter_files}")
            adapter_files_found = True
        
        # Check if any checkpoint files exist
        checkpoint_files = [f for f in files if '.pt' in f or 'checkpoint' in f]
        if checkpoint_files:
            print(f"- Checkpoint files: {checkpoint_files}")
            
            # Examine first checkpoint file to see if it has adapter weights
            if checkpoint_files:
                checkpoint_path = location / checkpoint_files[0]
                print(f"\nExamining checkpoint: {checkpoint_path}")
                
                extract_cmd = [
                    PYTHON_PATH, "-c",
                    f"""
import torch
try:
    checkpoint = torch.load("{checkpoint_path}", map_location="cpu")
    print(f"Checkpoint type: {{type(checkpoint)}}")
    
    if isinstance(checkpoint, dict):
        print(f"Keys: {{list(checkpoint.keys())}}")
        
        if "state" in checkpoint:
            state = checkpoint["state"]
            print(f"State keys: {{list(state.keys())}}")
            
            if "model" in state:
                model_state = state["model"]
                lora_keys = [k for k in model_state.keys() if 'lora_' in k]
                print(f"Found {{len(lora_keys)}} LoRA keys in checkpoint")
                if lora_keys:
                    print(f"Sample LoRA keys: {{lora_keys[:3]}}")
                    
                    # Extract and save adapter weights
                    lora_state = {{k: v for k, v in model_state.items() if 'lora_' in k}}
                    torch.save(lora_state, "{run_folder}/extracted_adapter_model.bin")
                    print(f"Extracted adapter weights saved to {{run_folder}}/extracted_adapter_model.bin")
except Exception as e:
    print(f"Error examining checkpoint: {{e}}")
"""
                ]
                subprocess.run(extract_cmd)
    
    if not adapter_files_found:
        print("\n❌ No adapter files found in expected locations")
        
    # Check debug log for any insights
    debug_log = "debug.log"
    if os.path.exists(debug_log):
        print("\nChecking debug log file:")
        with open(debug_log, "r") as f:
            log_content = f.read()
        
        debug_lines = [line for line in log_content.split('\n') if '[DEBUG]' in line]
        print(f"Found {len(debug_lines)} debug lines")
        for line in debug_lines[-30:]:  # Last 30 debug lines
            print(line)
    
    MODEL_CHECKPOINT_VOLUME.commit()
    return str(run_folder)

###########################
@app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              max_containers=1)
def extract_adapter_from_best_checkpoint(checkpoint_dir: str):
    """Extract adapter from the most recent/best checkpoint available"""
    import os, torch, json
    from pathlib import Path
    
    # Convert to Path without reassigning the parameter
    dir_path = Path(checkpoint_dir)
    print(f"Looking for checkpoints in: {dir_path}")
    
    # First try latest-rank0.pt
    latest_path = dir_path / "latest-rank0.pt"
    if latest_path.exists():
        print(f"Found latest-rank0.pt - trying this first")
        try:
            checkpoint = torch.load(latest_path, map_location="cpu")
            print("Successfully loaded latest-rank0.pt")
            checkpoint_path = latest_path
        except Exception as e:
            print(f"Error loading latest-rank0.pt: {e}")
            checkpoint = None
            checkpoint_path = None
    else:
        print("latest-rank0.pt not found")
        checkpoint = None
        checkpoint_path = None
    
    # If latest didn't work, find epoch checkpoints
    if checkpoint is None:
        print("Looking for epoch-batch checkpoints...")
        checkpoint_files = sorted(dir_path.glob("ep*-ba*-rank*.pt"))
        
        if not checkpoint_files:
            return "No checkpoint files found"
        
        # Sort by batch number to get the latest
        checkpoint_files.sort(key=lambda x: int(str(x).split('-ba')[1].split('-')[0]), reverse=True)
        checkpoint_path = checkpoint_files[0]
        
        print(f"Using latest epoch checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            print(f"Successfully loaded {checkpoint_path.name}")
        except Exception as e:
            return f"Error loading checkpoint {checkpoint_path.name}: {e}"
    
    # Extract adapter weights
    if "state" in checkpoint and "model" in checkpoint["state"]:
        model_state = checkpoint["state"]["model"]
        lora_state = {k: v for k, v in model_state.items() if "lora_" in k}
        if lora_state:
            print(f"Found {len(lora_state)} LoRA weights")
            sample_keys = list(lora_state.keys())[:3]
            print(f"Sample keys: {sample_keys}")
            
            # Save adapter files
            output_dir = Path(checkpoint_dir).parent  # Go up one level from native_checkpoints
            config_path = output_dir / "adapter_config.json"
            weights_path = output_dir / "adapter_model.bin"
            
            # Create adapter config
            config_dict = {
                "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 8,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "inference_mode": False
            }
            
            # Save files
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
                
            torch.save(lora_state, weights_path)
            
            print(f"✅ Successfully saved adapter files:")
            print(f"  - Config: {config_path}")
            print(f"  - Weights: {weights_path} ({os.path.getsize(weights_path)/1024/1024:.2f} MB)")
            
            # Verify the files exist
            if os.path.exists(config_path) and os.path.exists(weights_path):
                return {
                    "success": True,
                    "adapter_dir": str(output_dir),
                    "config_path": str(config_path),
                    "weights_path": str(weights_path),
                    "checkpoint_used": str(checkpoint_path),
                    "num_weights": len(lora_state)
                }
            else:
                return "Failed to verify adapter files were created"
        else:
            return "No LoRA weights found in checkpoint"
    else:
        return "Checkpoint doesn't have the expected structure"


@app.local_entrypoint()
def main():
    import time
    # print("Fixing adapter parameters...")
    # result = fix_adapter_params.remote()
    # print(f"Result: {result}")
    # time.sleep(2)
    
    # Generate a timestamp for this run
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Train a model first to create checkpoints
    print("Starting training...")
    model_path = train_with_hf_token.remote(run_ts)
    print(f"Training completed, folder: {model_path}")
    
    # Step 2: Extract adapters from the checkpoint we just created
    # Use the exact path returned from training
    checkpoint_path = f"{model_path}/native_checkpoints/latest-rank0.pt"
    
    print(f"Extracting adapters from checkpoint: {checkpoint_path}")
    adapter_dir = extract_adapter_from_best_checkpoint.remote(checkpoint_path)
    
    if adapter_dir:
        print(f"Success! Adapters extracted to: {adapter_dir}")
    else:
        print("Failed to extract adapters")
    
    return "Training and adapter extraction completed"



    # # Check for adapter files
    # adapter_config = Path(run_folder) / "adapter_config.json"
    # adapter_weights = Path(run_folder) / "adapter_model.bin"
    
    # print(f"Checking for adapter files:")
    # print(f"- Config exists: {adapter_config.exists()}")
    # print(f"- Weights exist: {adapter_weights.exists()}")
    
    # # If files aren't where we expect, search everywhere
    # if not (adapter_config.exists() and adapter_weights.exists()):
    #     print("Adapter files not found in expected location. Searching...")
    #     for root, dirs, files in os.walk(run_folder):
    #         adapter_files = [f for f in files if 'adapter' in f]
    #         if adapter_files:
    #             print(f"Found adapter files in {root}:")
    #             for f in adapter_files:
    #                 print(f"  - {f}")
    #             # Try to move them to the expected location
    #             for f in adapter_files:
    #                 src = os.path.join(root, f)
    #                 dst = os.path.join(run_folder, f)
    #                 try:
    #                     shutil.copy(src, dst)
    #                     print(f"Copied {f} to {run_folder}")
    #                 except Exception as e:
    #                     print(f"Failed to copy {f}: {e}")
    
    # MODEL_CHECKPOINT_VOLUME.commit()
    # return str(run_folder)


# def extract_adapters_from_checkpoint(checkpoint_path):
#     """Extract adapter weights from an existing checkpoint file"""
#     import os, torch, json
#     from pathlib import Path
    
#     print(f"Extracting adapters from checkpoint: {checkpoint_path}")
    
#     # Load checkpoint
#     checkpoint = torch.load(checkpoint_path, map_location="cpu")
#     print(f"Checkpoint loaded, type: {type(checkpoint)}")
    
#     # Navigate to model weights
#     if "state" in checkpoint and "model" in checkpoint["state"]:
#         model_state = checkpoint["state"]["model"]
        
#         # Extract LoRA weights
#         lora_weights = {k: v for k, v in model_state.items() if "lora_" in k}
#         print(f"Found {len(lora_weights)} LoRA weights in checkpoint")
        
#         if lora_weights:
#             # Determine save location - use parent directory of checkpoint
#             checkpoint_file = Path(checkpoint_path)
#             save_dir = checkpoint_file.parent.parent  # Go up two levels
            
#             # Create adapter config
#             adapter_config = {
#                 "base_model_name_or_path": "meta-llama/Llama-3.2-1B",
#                 "peft_type": "LORA",
#                 "task_type": "CAUSAL_LM",
#                 "r": 8,
#                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
#                 "lora_alpha": 16,
#                 "lora_dropout": 0.05,
#                 "inference_mode": False
#             }
            
#             # Save adapter files
#             config_path = save_dir / "adapter_config.json"
#             weights_path = save_dir / "adapter_model.bin"
            
#             with open(config_path, "w") as f:
#                 json.dump(adapter_config, f, indent=2)
                
#             torch.save(lora_weights, weights_path)
            
#             print(f"✅ Successfully extracted adapter files:")
#             print(f"  - Config: {config_path}")
#             print(f"  - Weights: {weights_path}")
            
#             # Verify the files exist
#             if os.path.exists(config_path) and os.path.exists(weights_path):
#                 print("Verified: Files were created successfully")
#             else:
#                 print("⚠️ Warning: Files were not created as expected")
            
#             return str(save_dir)
#         else:
#             print("❌ No LoRA weights found in checkpoint")
#     else:
#         print("❌ Checkpoint structure not as expected")
    
#     return None




# @app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
#               secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
#                       DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
#               max_containers=1)
# def debug_adapter_creation(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml"):
#     """Debug why adapter files aren't being created."""
#     import os, subprocess, shutil, time, yaml, json
#     from pathlib import Path
    
#     # Change to llm-foundry/scripts directory
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     # Get HF token
#     hf_token = os.environ.get("huggingface_secret_HF_TOKEN", "")
    
#     # 1. First, check dataset existence
#     data_path = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
#     index_path = Path(data_path) / "train_small" / "index.json"
    
#     if not index_path.exists():
#         print(f"Dataset not found at {index_path}, converting dataset...")
#         data_cmd = [
#             PYTHON_PATH,
#             "data_prep/convert_dataset_hf.py",
#             "--dataset", "allenai/c4",
#             "--data_subset", "en",
#             "--out_root", data_path,
#             "--splits", "train_small", "val_small",
#             "--concat_tokens", "2048",
#             "--tokenizer", "meta-llama/Llama-3.2-1B",
#         ]
#         subprocess.run(data_cmd, check=True)
#         print("Dataset converted successfully")
    
#     # 2. Set up model and download model files
#     print("\nPre-downloading model files...")
#     download_script = f"""
# import os
# from huggingface_hub import snapshot_download, login

# token = "{hf_token}"
# os.environ["HF_TOKEN"] = token
# login(token=token)

# local_dir = "/tmp/llama-3-2-1b"
# print(f"Downloading model to {{local_dir}}")
# snapshot_download(
#     repo_id="meta-llama/Llama-3.2-1B",
#     local_dir=local_dir,
#     token=token,
#     local_dir_use_symlinks=False
# )
# print("Download complete!")
# """
    
#     with open("/tmp/download_model.py", "w") as f:
#         f.write(download_script)
    
#     subprocess.run([PYTHON_PATH, "/tmp/download_model.py"])
    
#     # 3. Set up model paths and folders
#     model_name = get_model_name(yaml_path)
#     run_folder = get_run_folder(run_ts, model_name)
#     save_folder = Path(f"{run_folder}/native_checkpoints")
#     save_folder.mkdir(exist_ok=True, parents=True)
    
#     # 4. Modify YAML to use local model files
#     with open(yaml_path) as f:
#         yaml_config = yaml.safe_load(f)
    
#     # Save original model name for reference
#     original_model = yaml_config['variables']['model_name_or_path']
    
#     # Replace with local path
#     yaml_config['variables']['model_name_or_path'] = "/tmp/llama-3-2-1b"
    
#     # Create a temporary YAML file
#     temp_yaml_path = "train/yamls/local_llama.yaml"
#     with open(temp_yaml_path, "w") as f:
#         yaml.dump(yaml_config, f)
    
#     print(f"Modified YAML: {original_model} → /tmp/llama-3-2-1b")
    
#     # 5. Debug our composer_llama_adapter.py
#     print("\n🔍 Examining composer_llama_adapter.py for debugging...")
#     adapter_file = "/llm-foundry/llmfoundry/models/llama/composer_llama_adapter.py"
    
#     # Read the file to check implementation
#     try:
#         with open(adapter_file, "r") as f:
#             adapter_code = f.read()
        
#         # Check for key methods that should be creating adapter files
#         has_on_save = "on_save_checkpoint" in adapter_code
#         has_get_state = "get_state_dict" in adapter_code
#         has_peft = "get_peft_model_state_dict" in adapter_code
        
#         print(f"Adapter code has:")
#         print(f"- on_save_checkpoint method: {has_on_save}")
#         print(f"- get_state_dict method: {has_get_state}")
#         print(f"- PEFT extraction: {has_peft}")
        
#         # Add extra debug prints to the adapter file
#         print("\nAdding debug prints to adapter file...")
#         debug_adapter_code = adapter_code.replace(
#             "def get_state_dict(self, state_dict=None):",
#             """def get_state_dict(self, state_dict=None):
#         print("\\n[DEBUG] get_state_dict called!")
#         import traceback
#         traceback.print_stack()"""
#         )
        
#         if has_on_save:
#             debug_adapter_code = debug_adapter_code.replace(
#                 "def on_save_checkpoint(self, checkpoint_path):",
#                 """def on_save_checkpoint(self, checkpoint_path):
#             print("\\n[DEBUG] on_save_checkpoint called with path:", checkpoint_path)"""
#             )
        
#         # Write the modified file back
#         with open(adapter_file, "w") as f:
#             f.write(debug_adapter_code)
#         print("Added debug prints to adapter file")
#     except Exception as e:
#         print(f"Error examining adapter file: {e}")
    
#     # 6. Instrumented training with VERBOSE logging
#     print("\n🔧 Running training with VERBOSE logging...")
    
#     # Set environment variable for adapter saving
#     os.environ["COMPOSER_SAVE_FOLDER"] = str(save_folder)
#     print(f"Set COMPOSER_SAVE_FOLDER={save_folder}")
    
#     # Run training
#     train_cmd = [
#         PYTHON_PATH,
#         "train/train_with_llama_adapter.py",
#         temp_yaml_path,
#         data_path,
#         f"save_folder={save_folder}",
#         f"max_duration={TRAIN_DURATION}",
#         f"save_interval={SAVE_INTERVAL}",
#         "save_latest_filename=latest-rank0.pt",
#         "model.should_save_peft_only=true",
#         "loggers.file.enabled=true",
#         "loggers.file.log_level=DEBUG",
#         "loggers.file.append=true",
#         "loggers.file.filename=adapter_debug.log"
#     ]
    
#     result = subprocess.run(train_cmd)
    
#     # 7. Check log file for clues
#     log_file = os.path.join(os.getcwd(), "adapter_debug.log")
#     if os.path.exists(log_file):
#         print("\n📜 Last 30 lines of debug log:")
#         with open(log_file, "r") as f:
#             lines = f.readlines()
#             for line in lines[-30:]:
#                 if "DEBUG" in line or "get_state_dict" in line or "on_save_checkpoint" in line:
#                     print(line.strip())
    
#     # 8. Check for checkpoint and adapter files
#     checkpoint_path = save_folder / "latest-rank0.pt"
#     print(f"\nChecking for checkpoint at {checkpoint_path}...")
    
#     if checkpoint_path.exists():
#         print("✅ Checkpoint exists")
#         import torch
#         try:
#             checkpoint = torch.load(checkpoint_path, map_location="cpu")
#             print(f"Checkpoint keys: {list(checkpoint.keys())}")
#             if "state" in checkpoint:
#                 print(f"State keys: {list(checkpoint['state'].keys())}")
#                 if "model" in checkpoint["state"]:
#                     model_dict = checkpoint["state"]["model"]
#                     keys = list(model_dict.keys())
#                     lora_keys = [k for k in keys if "lora_" in k]
#                     print(f"Found {len(lora_keys)}/{len(keys)} LoRA keys in checkpoint")
#                     if lora_keys:
#                         print("Example LoRA keys:", lora_keys[:3])
#         except Exception as e:
#             print(f"Error loading checkpoint: {e}")
#     else:
#         print("❌ Checkpoint does not exist")
    
#     adapter_config_path = Path(run_folder) / "adapter_config.json"
#     adapter_weights_path = Path(run_folder) / "adapter_model.bin"
    
#     print(f"\nChecking for adapter files...")
#     print(f"- adapter_config.json: {adapter_config_path.exists()}")
#     print(f"- adapter_model.bin: {adapter_weights_path.exists()}")
    
#     # 9. Full directory listing for debugging
#     print("\nDirectory contents for debugging:")
#     for path in [run_folder, save_folder]:
#         if path.exists():
#             print(f"\nContents of {path}:")
#             for item in os.listdir(path):
#                 print(f"  {item}")
    
#     # 10. Report conclusion
#     if adapter_config_path.exists() and adapter_weights_path.exists():
#         print("\n✅ SUCCESS: Adapter files were created!")
#     else:
#         print("\n❌ FAILURE: Adapter files were not created")
#         raise ValueError("No adapter files found - debugging needed for composer_llama_adapter.py!")
    
#     MODEL_CHECKPOINT_VOLUME.commit()
#     return str(run_folder)

# @app.function(image=image)
# def check_adapter_file():
#     """Print the contents of composer_llama_adapter.py for inspection"""
#     adapter_file = "/llm-foundry/llmfoundry/models/llama/composer_llama_adapter.py"
    
#     with open(adapter_file, "r") as f:
#         content = f.read()
        
#     print(f"Adapter file contents ({len(content)} bytes):")
#     print(content[:1000] + "..." if len(content) > 1000 else content)
    
#     return "Adapter file checked"
# @app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
#               secrets=[Secret.from_name("LRG")],
#               volumes={DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
#               max_containers=1)
# def check_dataset():
#     """Just check if dataset exists without trying to recreate."""
#     import os
#     from pathlib import Path
    
#     # Change to llm-foundry/scripts directory
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     # Check dataset state
#     data_path = Path(f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small")
#     train_index = data_path / "train_small" / "index.json"
#     val_index = data_path / "val_small" / "index.json"
    
#     print(f"Checking dataset at {data_path}")
#     print(f"- Train index exists: {train_index.exists()}")
#     print(f"- Val index exists: {val_index.exists()}")
    
#     # Check what's in the directory
#     print("\nDirectory contents:")
#     for item in os.listdir(data_path):
#         item_path = os.path.join(data_path, item)
#         if os.path.isdir(item_path):
#             files = os.listdir(item_path)
#             file_count = len(files)
#             print(f"- {item}: {file_count} files {'(including index.json)' if 'index.json' in files else ''}")
    
#     if train_index.exists() and val_index.exists():
#         print("\n✅ Complete dataset found!")
#     else:
#         print("\n⚠️ Dataset incomplete, but we'll use what's available")
    
#     return str(data_path)
# @app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
#               secrets=[Secret.from_name("LRG")],
#               volumes={DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
#               max_containers=1)
# def cleanup_dataset():
#     """Clean up corrupted dataset and create a fresh one."""
#     import os, subprocess, shutil
#     from pathlib import Path
    
#     # Change to llm-foundry/scripts directory
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     # Check current dataset state
#     data_path = Path(f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small")
#     print(f"Examining dataset at {data_path}")
    
#     if data_path.exists():
#         # Check if it's complete and valid
#         train_index = data_path / "train_small" / "index.json"
#         val_index = data_path / "val_small" / "index.json"
        
#         if train_index.exists() and val_index.exists():
#             print("✅ Dataset appears to be complete and valid, no cleanup needed")
#             return str(data_path)
#         else:
#             print("❌ Dataset is incomplete or corrupted, will remove and recreate")
            
#             # Backup the old data just in case
#             print("Making backup of existing data...")
#             backup_dir = Path(f"{DATASETS_VOLUME_MOUNT_PATH}/c4_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
#             backup_dir.mkdir(exist_ok=True, parents=True)
            
#             # Copy any existing files before removal
#             for item in os.listdir(data_path):
#                 src = data_path / item
#                 dst = backup_dir / item
#                 try:
#                     if os.path.isdir(src):
#                         shutil.copytree(src, dst)
#                     else:
#                         shutil.copy2(src, dst)
#                 except Exception as e:
#                     print(f"Warning during backup: {e}")
            
#             # Remove the corrupted dataset
#             try:
#                 shutil.rmtree(data_path)
#                 print(f"Removed corrupted dataset at {data_path}")
#             except Exception as e:
#                 print(f"Error removing dataset: {e}")
#                 # If we can't remove, rename it
#                 try:
#                     old_path = Path(f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small_corrupted")
#                     shutil.move(data_path, old_path)
#                     print(f"Renamed corrupted dataset to {old_path}")
#                 except Exception as e2:
#                     print(f"Error renaming dataset: {e2}")
#                     return "Failed to clean up dataset"
    
#     # Create fresh directory
#     data_path.mkdir(exist_ok=True, parents=True)
    
#     # Create a fresh dataset
#     print("Creating fresh dataset...")
#     data_cmd = [
#         PYTHON_PATH,
#         "data_prep/convert_dataset_hf.py",
#         "--dataset", "allenai/c4", 
#         "--data_subset", "en",
#         "--out_root", str(data_path),
#         "--splits", "train_small", "val_small",
#         "--concat_tokens", "2048", 
#         "--tokenizer", "meta-llama/Llama-3.2-1B"    ]
    
#     try:
#         subprocess.run(data_cmd, check=True)
#         print("✅ Dataset created successfully!")
        
#         # Double check it was created correctly
#         train_index = data_path / "train_small" / "index.json"
#         val_index = data_path / "val_small" / "index.json"
        
#         if train_index.exists() and val_index.exists():
#             print("✓ Verified: index.json files exist")
#         else:
#             print("⚠️ Warning: index.json files missing after creation")
        
#         # Commit volume changes
#         DATASETS_VOLUME.commit()
#         return str(data_path)
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to create dataset: {e}")
#         return "Dataset creation failed"




#@app.function(image=image)
# def fix_adapter_params():
#     """Fix parameter mismatch in CustomLlamaModel"""
#     import os
    
#     # Path to adapter file
#     adapter_file = "/llm-foundry/llmfoundry/models/llama/composer_llama_adapter.py"
    
#     # Read current file
#     with open(adapter_file, "r") as f:
#         content = f.read()
    
#     # Fix the super().__init__ call by removing the 'config' parameter
#     fixed_init = '''super().__init__(
#             pretrained_model_name_or_path=pretrained_model_name_or_path,
#             tokenizer=tokenizer,
#             use_logits=use_logits,
#             metrics=metrics,
#             eval_metrics=eval_metrics,
#             shift_labels=shift_labels,
#             allow_embedding_resizing=allow_embedding_resizing,
#             init_device=init_device,
#             **kwargs
#         )'''
    
#     # Replace the super().__init__ call in the content
#     if "super().__init__(" in content:
#         import re
#         # Use regex to find and replace the entire super().__init__(...) call
#         pattern = r"super\(\)\.__init__\([^)]*\)"
#         content = re.sub(pattern, fixed_init, content)
        
#         # Write the fixed code
#         with open(adapter_file, "w") as f:
#             f.write(content)
            
#         return "Adapter params fixed - removed 'config' parameter"
#     else:
#         return "Could not find super().__init__ call to fix"

# @app.local_entrypoint()
# def main():
#     from pathlib import Path
#     import time, os
#     run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

#     # Step 1: Check system stats
#     get_stats.remote()
#     time.sleep(1)
#     convert_c4_small_dataset.remote() # Only run once
#     # Step 2: Train with AIM visualization
#     yaml_path = "train/yamls/llama/llama3-1b-lora2.yaml"
#     model_path = train_model.remote(run_ts, yaml_path=yaml_path,hf_token= os.environ.get("HF_TOKEN"))
#     print(f"Model path: {model_path}")
#     time.sleep(1)
    
#     # Step 3: Convert to HuggingFace format
#     hf_model_path = convert_model_to_hf.remote(model_path)
#     time.sleep(1)
