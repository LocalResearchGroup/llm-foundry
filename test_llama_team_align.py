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


@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
              secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],  # Add HF secret
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                      DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
    max_containers=1
)
def train_model(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml",
                hf_token: str = ''):
    import os, subprocess, shutil, time, yaml
    from pathlib import Path
    
    # Change to llm-foundry/scripts directory
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    # Set HF token
    if hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        print("HF token set in environment variables")
    
    # Pre-download the model files
    print("\n🔄 Pre-downloading Llama model files...")
    download_script = f"""
import os
from huggingface_hub import snapshot_download, login

# Set token
token = "{hf_token}"
os.environ["HF_TOKEN"] = token
login(token=token)

# Download model files
local_dir = "/tmp/llama-3-2-1b"
print(f"Downloading model to {{local_dir}}")
snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B",
    local_dir=local_dir,
    token=token,
    local_dir_use_symlinks=False
)
print("Download complete!")
"""
    
    with open("/tmp/download_model.py", "w") as f:
        f.write(download_script)
    
    subprocess.run([PYTHON_PATH, "/tmp/download_model.py"])
    
    # Modify YAML to use the local model
    print("\n📝 Modifying YAML to use local model files...")
    with open(yaml_path) as f:
        yaml_config = yaml.safe_load(f)
    
    # Save original model name for reference
    original_model = yaml_config['variables']['model_name_or_path']
    
    # Replace with local path
    yaml_config['variables']['model_name_or_path'] = "/tmp/llama-3-2-1b"
    
    # Create a temporary YAML file
    temp_yaml_path = "train/yamls/local_llama.yaml"
    with open(temp_yaml_path, "w") as f:
        yaml.dump(yaml_config, f)
    
    print(f"Modified YAML: {original_model} → /tmp/llama-3-2-1b")
    
    # Continue with your existing code...
    model_name = get_model_name(yaml_path)
    run_folder = get_run_folder(run_ts, model_name)
    save_folder = Path(f"{run_folder}/native_checkpoints")
    
    # Ensure directory exists
    save_folder.mkdir(exist_ok=True, parents=True)
    shutil.copy(temp_yaml_path, save_folder / Path(yaml_path).name)
    
    # Set environment variable for save folder - used by our custom adapter
    os.environ["COMPOSER_SAVE_FOLDER"] = str(save_folder)
    print(f"Set COMPOSER_SAVE_FOLDER={save_folder} for adapter saving")
    
    # Run training with the modified YAML
    data_path = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
    train_cmd = [
        PYTHON_PATH,
        "train/train_with_llama_adapter.py",
        temp_yaml_path,  # Use our modified YAML with local path
        data_path,
        f"save_folder={save_folder}",
        f"max_duration={TRAIN_DURATION}",
        f"save_interval={SAVE_INTERVAL}",
        "save_latest_filename=latest-rank0.pt",
        "model.should_save_peft_only=true"
    ]
    
    print(f"Running training command...")
    result = subprocess.run(train_cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")
    # Wait a moment to ensure all files are saved
    time.sleep(2)
    
    # Check if adapter files were created by our custom adapter class
    adapter_config_path = Path(run_folder) / "adapter_config.json"
    adapter_weights_path = Path(run_folder) / "adapter_model.bin"
    
    print("\nChecking for adapter files...")
    if adapter_config_path.exists() and adapter_weights_path.exists():
        print(f"✅ Found adapter files created by our custom adapter:")
        print(f"   - Config: {adapter_config_path}")
        print(f"   - Weights: {adapter_weights_path}")
        
        # Check if adapter weights are non-empty
        import torch
        try:
            adapter_weights = torch.load(adapter_weights_path, map_location="cpu")
            if adapter_weights:
                print(f"✅ Adapter weights contain {len(adapter_weights)} parameters")
                # Show a sample key
                sample_keys = list(adapter_weights.keys())[:2]
                print(f"   Sample keys: {sample_keys}")
            else:
                
                print("⚠️ WARNING: Adapter weights file is empty! Training may not have worked.")
        except Exception as e:
            print(f"⚠️ Error loading adapter weights: {e}")
    else:
        print("\n⚠️ Adapter files not found! Debugging information:")
        print(f"Directory contents of {run_folder}:")
        for item in os.listdir(run_folder):
            print(f"  {item}")
            
        print(f"\nDirectory contents of {save_folder}:")
        for item in os.listdir(save_folder):
            print(f"  {item}")
        
        # Show log tail from training
        print("\nChecking for any checkpoint related logs in the training output...")
        # Look for any files in save_folder/checkpoints
        checkpoint_dir = save_folder / "checkpoints"
        if checkpoint_dir.exists():
            print(f"Found checkpoint directory at {checkpoint_dir}")
            print("Contents:")
            for item in os.listdir(checkpoint_dir):
                print(f"  {item}")
        raise ValueError("⚠️ No adapter files found, creating them manually...")

    # Save tokenizer explicitly (this is always useful)
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        tokenizer.save_pretrained(run_folder)
        print(f"Saved tokenizer to {run_folder}")
    except Exception as e:
        print(f"Error saving tokenizer: {e}")
    
    # Check for standard checkpoint file
    checkpoint_path = save_folder / "latest-rank0.pt"
    if not checkpoint_path.exists():
        raise ValueError("⚠️ No checkpoint found from training!")
    else:
        print(f"✅ Found checkpoint at {checkpoint_path}")
    
    # Make sure changes are committed to the volume
    MODEL_CHECKPOINT_VOLUME.commit()
    print(f'Training complete for {run_ts}')
    print(f'Model checkpoints saved to {save_folder}')
    
    return str(run_folder)


@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              concurrency_limit=1)
def convert_model_to_hf(checkpoint_path: str, upload_to_hf: bool = False):
    """Convert a model checkpoint to a HuggingFace format."""
    import subprocess, os
    from pathlib import Path
    env = os.environ.copy()
    env["IS_PEFT"] = "True"  # Set PEFT flag
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
        "--is_peft",f"{IS_PEFT}",
        "--train_yaml",f"{TRAIN_YAML}"
    ]
    if upload_to_hf: convert_cmd.extend(["--hf_repo_for_upload", f"LocalResearchGroup/{run_folder.name}"])

    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Conversion errors:", result.stderr)
    # Verify all required files are present
    print("\nVerifying required files for evaluation...")
    required_files = ["config.json", "tokenizer_config.json", "pytorch_model.bin"]
    for file in required_files:
        path = os.path.join(hf_output_path, file)
        print(f"{file}: {'✅ Present' if os.path.exists(path) else '❌ Missing'}")
    
    MODEL_CHECKPOINT_VOLUME.commit()
    print("Conversion complete!")



@app.local_entrypoint()
def main():
    from pathlib import Path
    import time, os
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Check system stats
    get_stats.remote()
    time.sleep(1)
    convert_c4_small_dataset.remote() # Only run once
    # Step 2: Train with AIM visualization
    yaml_path = "train/yamls/llama/llama3-1b-lora2.yaml"
    model_path = train_model.remote(run_ts, yaml_path=yaml_path,hf_token= os.environ.get("HF_TOKEN"))
    print(f"Model path: {model_path}")
    time.sleep(1)
    
    # Step 3: Convert to HuggingFace format
    hf_model_path = convert_model_to_hf.remote(model_path)
    time.sleep(1)

    #TURN OFF EVAL FOR NOW    
    # Step 4: Evaluate model
    # evaluate_model.remote(model_path)
    # time.sleep(1)
    
    # push_folder_to_hf.remote(Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/model_path) 
    # time.sleep(1)

    # Step 5: Generate responses
    #generate_responses.remote(model_path)
    
    return "Llama training and evaluation pipeline completed!"




### IGNORE:
# def run_aim_server(run_folder: str):
#     import os, subprocess
#     from pathlib import Path
    
#     Path(run_folder).mkdir(exist_ok=True)
#     pwd = os.getcwd()
#     os.chdir(run_folder)
#     print("Initializing Aim...")
#     subprocess.run(["aim", "init"], check=True)
    
#     # Background process that needs to be closed by calling function using .terminate()
#     process = subprocess.Popen(
#         ["aim", "up", "--host", "0.0.0.0", "--port", "43800"],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#     )
#     os.chdir(pwd)
#     return process


# @app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, 
#               secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],  # Add HF secret
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
#                       DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
#     max_containers=1
# )
# def train_with_aim(run_ts: str, yaml_path: str = "train/yamls/llama/llama3-1b-lora2.yaml"):
#     """Train model with AIM visualization"""
#     import subprocess, time, os
    
#     ##########
#     # Debug what data files exist
#     print("\nChecking data directory structure:")
#     data_base = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
#     if os.path.exists(data_base):
#         print(f"Base data directory {data_base} exists")
#         # List its contents
#         print("Contents:")
#         for item in os.listdir(data_base):
#             print(f"  {item}")
#             if os.path.isdir(f"{data_base}/{item}"):
#                 print(f"    Contents of {item}:")
#                 try:
#                     for subitem in os.listdir(f"{data_base}/{item}"):
#                         print(f"      {subitem}")
#                 except:
#                     print("      (Error listing directory)")
#     else:
#         print(f"Data directory {data_base} doesn't exist!")
#         print("Will need to run data conversion first")
#         # Run data conversion directly
#         subprocess.run([
#             PYTHON_PATH,
#             "/llm-foundry/scripts/data_prep/convert_dataset_hf.py",
#             "--dataset", "allenai/c4",
#             "--data_subset", "en",
#             "--out_root", data_base,
#             "--splits", "train_small", "val_small",
#             "--concat_tokens", "2048",
#             "--tokenizer", "meta-llama/Llama-3.2-1B"
#         ], check=True)
#         print(f"Data conversion completed, checking directory again:")
#         if os.path.exists(data_base):
#             print("Contents after conversion:")
#             for item in os.listdir(data_base):
#                 print(f"  {item}")
    

    # ##########
    # # First prepare dataset
    # #convert_c4_dataset.remote()
    
    # with modal.forward(43800) as tunnel:
    #     print(f"\nAim server available at: {tunnel.url}")
    #     model_path = None
    #     aim_task = run_aim_server(get_run_folder(run_ts, get_model_name(yaml_path)))
    #     time.sleep(5)
    
    #     try:
    #         hf_token = get_hf_token()
    #         model_path = train_model(run_ts, yaml_path,hf_token)
    #     finally:
    #         aim_task.terminate()
    #         try:
    #             aim_task.wait(timeout=5)
    #         except subprocess.TimeoutExpired:
    #             aim_task.kill()
    
    # return model_path



# @app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
#               max_containers=1)
# def push_folder_to_hf(folder_path: str, repo_id: str | None = None, repo_type: str = "model", private: bool = True):
#     """Upload model checkpoint to HuggingFace Hub."""
#     from huggingface_hub import HfApi
#     from pathlib import Path
#     # Set up authentication
#     token = setup_hf_auth()
#     if not token:
#         print("ERROR: HuggingFace token not found, cannot push to hub")
#         return
#     folder_path = Path(folder_path)
#     if not folder_path.exists() or not folder_path.is_dir():
#         raise FileNotFoundError(f"Folder {folder_path} does not exist or is not a directory.")
#     folder_name = folder_path.name
#     if repo_id is None: repo_id = f"LocalResearchGroup/{folder_name}"

#     api = HfApi()

#     print(f'Uploading {folder_path} to HuggingFace Hub at {repo_id}')
    
#     api.create_repo(repo_id=repo_id, use_auth_token=True, repo_type=repo_type, private=private, exist_ok=True)
#     print('Repo created.')

#     api.upload_folder(folder_path=folder_path, repo_id=repo_id, use_auth_token=True, repo_type=repo_type)
#     print(f'Folder "{folder_path}" uploaded to: "{repo_id}" successfully.')


# @app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
#               secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
#               max_containers=1)
# def evaluate_model(checkpoint_path: str):
#     """Evaluate PEFT/LoRA model using direct PEFT loading"""
#     import subprocess, os, time
#     from pathlib import Path
    
#     # Ensure HF authentication
#     setup_hf_auth()
    
#     os.chdir("/llm-foundry/scripts")
#     print(f"Working directory: {os.getcwd()}")
    
#     model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
#     save_path = model_path/"evals"
#     save_path.mkdir(exist_ok=True, parents=True)
    
#     # Instead of complex string formatting, use a simple Python command to run evaluation
#     print("\nRunning PEFT evaluation directly...")
    
#     eval_cmd = [
#         PYTHON_PATH, "-c",
#         f"""
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, PeftConfig
# import json, os
# import pandas as pd

# # Use paths directly from command line
# model_path = "{model_path}"
# save_path = "{save_path}"

# print(f"Evaluating model: {model_path}")

# # Load tokenizer
# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# # Load base model
# print("Loading base model...")
# base_model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.2-1B", 
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )

# # Load adapter
# print("Loading adapter...")
# model = PeftModel.from_pretrained(base_model, model_path)

# # Evaluation examples
# examples = [
#     "The capital of France is",
#     "Explain quantum computing in simple terms:",
#     "Write a short poem about machine learning:",
#     "What are the three laws of robotics?",
#     "Describe the process of photosynthesis:"
# ]

# print("Running evaluation...")
# results = []
# for i, example in enumerate(examples):
#     print(f"Example {{i+1}}/{{len(examples)}}")
#     inputs = tokenizer(example, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs.input_ids,
#             max_new_tokens=100,
#             temperature=0.1,
#             do_sample=True
#         )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     completion = response[len(example):]
#     results.append({{"prompt": example, "response": completion}})
    
#     print(f"Prompt: {{example}}")
#     print(f"Response: {{completion[:100]}}...")

# # Save results
# print("Saving results...")
# with open(os.path.join(save_path, "peft_eval_results.json"), "w") as f:
#     json.dump(results, f, indent=2)

# print("Evaluation completed!")
#         """
#     ]
    
#     result = subprocess.run(eval_cmd)
    
#     # Check if the results were generated
#     results_path = save_path / "peft_eval_results.json"
    
#     if results_path.exists():
#         print(f"✅ Evaluation results saved to {results_path}")
#         # Display a summary
#         try:
#             with open(results_path, "r") as f:
#                 import json
#                 results = json.load(f)
#                 print(f"Generated {len(results)} responses")
#         except Exception as e:
#             print(f"Error reading results: {e}")
#     else:
#         print(f"⚠️ No evaluation results found at {results_path}")
    
#     MODEL_CHECKPOINT_VOLUME.commit()
#     print("Evaluation complete!")
#     return str(save_path)
# @app.function(gpu=TRAINING_GPU, image=image, timeout=3600,
#               secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],  # Add HF secret
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
#               max_containers=1)
# def generate_responses(checkpoint_path: str, prompts: list[str]|str|None=None):
#     import subprocess, os
#     from pathlib import Path
#     setup_hf_auth()
#     os.chdir("/llm-foundry/scripts")
    
#     model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path

#     if prompts is None:
#         prompts = [
#             "The answer to life, the universe, and happiness is",
#             "Here's a quick recipe for baking chocolate chip cookies: Start by",
#         ]
#     elif isinstance(prompts, str):
#         prompts = [prompts]
    
#     # Use direct PEFT evaluation
#     eval_cmd = [
#         PYTHON_PATH, "-c", f"""
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# import json

# model_path = "{model_path}"
# prompts = {prompts}

# print("\\nGenerating responses using PEFT model...")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# base_model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.2-1B", 
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )
# model = PeftModel.from_pretrained(base_model, model_path)

# results = []
# for prompt in prompts:
#     print(f"\\nPrompt: {{prompt}}")
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     outputs = model.generate(
#         inputs.input_ids,
#         max_new_tokens=100,
#         do_sample=True,
#         temperature=0.7
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     completion = response[len(prompt):]
#     print(f"Response: {{completion}}")
#     results.append({{"prompt": prompt, "response": completion}})

# with open("{model_path}/generations.json", "w") as f:
#     json.dump(results, f, indent=2)
# print("\\nGeneration complete!")
# """
#     ]
    
#     subprocess.run(eval_cmd)
#     print("Generation complete!")
#     return f"{model_path}/generations.json"

