import modal
from modal import Image, App, Secret, Volume
from omegaconf import OmegaConf
import pathlib, datetime

# Constants
PYTHON_PATH = "/opt/conda/envs/llm-foundry/bin/python"
TRAINING_GPU = "a100"  # "l4"
BATCH_SIZE = 16  # Adjusted for your model
TRAIN_DURATION="2ba"#"500ba"
EVAL_INTERVAL="100ba"#"100ba"
SAVE_INTERVAL="100ba"#"100ba"
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

def train_model(run_ts: str, yaml_path: str = "train/yamls/pretrain/smollm2-135m.yaml",
                hf_token: str = ''):
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

    save_folder.mkdir(exist_ok=True, parents=True)
    shutil.copy(yaml_path, Path(save_folder) / Path(yaml_path).name)
    
    # Use a consistent data path for both approaches
    data_path = f"{DATASETS_VOLUME_MOUNT_PATH}/c4_small"
    global USE_CUSTOM_MODEL
    if USE_CUSTOM_MODEL and hf_token:
        # Use explicit token passed from parent function
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        print("Using HF token passed from parent function")
        
        # Also log in using the CLI for broader compatibility
        login_cmd = ["huggingface-cli", "login", "--token", hf_token]
        try:
            subprocess.run(login_cmd, check=True, capture_output=True)
            print("Logged in to HuggingFace using CLI")
        except Exception as e:
            print(f"Warning: HF CLI login failed: {e}")
            
        # train_cmd = [
        #     PYTHON_PATH,
        #     "train/train_with_llama_adapter.py",
        #     yaml_path, 
        #     data_path
        # ]
        train_cmd = [
        PYTHON_PATH,
        "train/train_with_llama_adapter.py",
        yaml_path, 
        data_path,
        f"loggers.aim.experiment_name=quickstart_{model_name}_modal",
        f"loggers.aim.repo={run_folder}/.aim",
        f"variables.data_local={data_path}",
        "train_loader.dataset.split=train_small",
        "eval_loader.dataset.split=val_small",
        f"max_duration={TRAIN_DURATION}",
        f"eval_interval={EVAL_INTERVAL}", 
        f"save_folder={save_folder}",
        f"save_interval={SAVE_INTERVAL}",
        f"device_eval_batch_size={BATCH_SIZE}",
        f"device_train_microbatch_size={BATCH_SIZE}",
        f"global_train_batch_size={BATCH_SIZE}",
        "save_latest_filename=latest-rank0.pt", #added
    ]

    else:
        print("\nUsing standard training approach...")
        train_cmd = [
            "composer",
            "train/train.py",
            yaml_path,
            f"loggers.aim.experiment_name=quickstart_{model_name}_modal",
            f"loggers.aim.repo={run_folder}/.aim",
            f"variables.data_local={data_path}",
            "train_loader.dataset.split=train_small",
            "eval_loader.dataset.split=val_small",
            f"max_duration={TRAIN_DURATION}",
            f"eval_interval={EVAL_INTERVAL}", 
            f"save_folder={save_folder}",
            f"save_interval={SAVE_INTERVAL}",
            f"device_eval_batch_size={BATCH_SIZE}",
            f"device_train_microbatch_size={BATCH_SIZE}",
            f"global_train_batch_size={BATCH_SIZE}",
        ]
    
    print(f"Running command: {' '.join(train_cmd)}")
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
    """Start AIM visualization server"""
    import os, subprocess
    from pathlib import Path
    
    Path(run_folder).mkdir(exist_ok=True)
    pwd = os.getcwd()
    os.chdir(run_folder)
    print("Initializing Aim...")
    subprocess.run(["aim", "init"], check=True)
    
    # Start AIM server
    process = subprocess.Popen(
        ["aim", "up", "--host", "0.0.0.0", "--port", "43800"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    os.chdir(pwd)
    return process

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

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
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
        "--output_precision", "bf16",
    ]
    if upload_to_hf: convert_cmd.extend(["--hf_repo_for_upload", f"LocalResearchGroup/{run_folder.name}"])
    env = os.environ.copy() #?
    env["IS_PEFT"] = "True" #?
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Conversion errors:", result.stderr)
    MODEL_CHECKPOINT_VOLUME.commit()
    print("Conversion complete!")


@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              max_containers=1)
def evaluate_model(checkpoint_path: str):
    import subprocess, os
    from pathlib import Path

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

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              max_containers=1)
def generate_responses(checkpoint_path: str, prompts: list[str]|str|None=None):
    import subprocess, os
    from pathlib import Path

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


@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
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
    
    # Step 4: Evaluate model
    evaluate_model.remote(model_path)
    time.sleep(1)
    
    push_folder_to_hf.remote(Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/model_path) 
    time.sleep(1)

    # Step 5: Generate responses
    generate_responses.remote(model_path)
    
    return "Llama training and evaluation pipeline completed!"