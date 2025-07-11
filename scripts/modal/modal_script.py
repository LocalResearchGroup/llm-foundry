#TODO:
# - Probably want to put local aim repo in the folder with the model checkpoints
# - Fix duplicate container issue

import os
import modal
import sys
from modal import Image, App, Secret, Volume

import pathlib, datetime

PYTHON_PATH = "/opt/conda/envs/llm-foundry/bin/python"

# command line arguments
TRAINING_GPU = os.environ.get("MODAL_GPU", "L4") 
TRAIN_YAML = os.environ.get("TRAIN_YAML", "")


IS_PEFT = os.environ.get("IS_PEFT", "True")
IS_PEFT = IS_PEFT in ("True", "true")

OUTPUT_PRECISION = os.environ.get("OUTPUT_PRECISION", "bf16")

# defaults --- make sure your Modal Volumes are titled accordingly
DATASET_BASE_PATH = "/datasets"
DATASETS_VOLUME = Volume.from_name("lrg-datasets", create_if_missing=True)
DATASETS_VOLUME_MOUNT_PATH = pathlib.Path("/datasets")
MODEL_CHECKPOINT_VOLUME = Volume.from_name("lrg-model-checkpoints", create_if_missing=True)
MODEL_CHECKPOINT_VOLUME_MOUNT_PATH = pathlib.Path("/model-checkpoints")

app = App("quick-start")

# Build image from local Dockerfile
image = Image.from_dockerfile("Dockerfile", gpu='L4')
image = image.add_local_file(TRAIN_YAML, f"/llm-foundry/scripts/train/yamls/finetune/{TRAIN_YAML}")

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


@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")], 
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
        "--tokenizer", "HuggingFaceTB/SmolLM2-135M",
        "--eos_text", "<|endoftext|>",
    ]
    result = subprocess.run(data_prep_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Data prep errors:", result.stderr)
    
    DATASETS_VOLUME.commit()
    
@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")], 
              volumes={DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def convert_finetuning_dataset():
    import subprocess
    import os
    
    DS_PATH = "meta-math/MetaMathQA"
    MODEL_HF_PATH = f"HuggingFaceTB/SmolLM2-135M"
    
    # Change to llm-foundry/scripts directory at the start
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Convert dataset
    print(f"Converting {DS_PATH} dataset...")
    data_prep_cmd = [
        PYTHON_PATH,
        "data_prep/convert_finetuning_dataset.py",
        "--dataset", DS_PATH,
        "--splits", "train",
        # Either make a preprocessor or use the predefined ones. 
        # The custom preprocessing function for meta-math/MetaMathQA is already defined in llmfoundry/data/finetuning/tasks.py
        # If you want to use a custom preprocessor in another folder, you can add the following line:
        # "--preprocessor", "path/to/preprocessor_file:preprocessor_function",
        "--out_root", f"{DATASET_BASE_PATH}/{DS_PATH}",
        "--tokenizer", MODEL_HF_PATH,
    ]

    subprocess.run(data_prep_cmd)
    
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

# @app.function(gpu=TRAINING_GPU, image=image, timeout=6*3600, secrets=[Secret.from_name("LRG")], 
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
#                        DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME})
def train_model(run_ts: str, yaml_path: str = "train/yamls/pretrain/smollm2-135m.yaml"):
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

    train_cmd = [
        "composer",
        "train/train.py",
        yaml_path, 
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


@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                      DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def train_with_aim(run_ts: str, yaml_path: str = "train/yamls/pretrain/smollm2-135m.yaml"):

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

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              max_containers=1)
def convert_model_to_hf(checkpoint_path: str, yaml_path: str = "", upload_to_hf: bool = False, is_peft: bool = IS_PEFT):
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

    print("Is peft:", is_peft)
    print("\nConverting model to HuggingFace format...")
    print(f"Train yaml: {TRAIN_YAML}")
    convert_cmd = [
        PYTHON_PATH, "inference/convert_composer_to_hf.py",
        "--composer_path", composer_checkpoint_path,
        "--hf_output_path", hf_output_path,
        "--output_precision", f"{OUTPUT_PRECISION}",
        "--is_peft", f"{is_peft}",
        "--train_yaml", f"{yaml_path}",
    ]
    if upload_to_hf: convert_cmd.extend(["--hf_repo_for_upload", f"LocalResearchGroup/{run_folder.name}"])

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

@app.function(gpu=TRAINING_GPU, image=image, timeout=10800, secrets=[Secret.from_name("LRG")],
              volumes={DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              concurrency_limit=1)
def pull_hf_to_folder():
    import subprocess
    import os
    
    # Change to llm-foundry/scripts directory at the start
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: pull all tokens
    print(f"Downloading repos to {DATASETS_VOLUME_MOUNT_PATH}/")
    data_prep_cmd = [
        PYTHON_PATH,  # Use the correct Python interpreter
        "data_prep/download_repo.py",
        "--out", f"{DATASETS_VOLUME_MOUNT_PATH}/",
    ]
    result = subprocess.run(data_prep_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Download data errors:", result.stderr)
    
    DATASETS_VOLUME.commit()

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              concurrency_limit=1)
def process_datasets():
    import subprocess
    import os
    
    # Change to llm-foundry/scripts directory at the start
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: pull all tokens
    print(f"Processing datasets...")
    data_prep_cmd = [
        PYTHON_PATH,  # Use the correct Python interpreter
        "data_prep/convert_dataset_hf.py",
    ]
    result = subprocess.run(data_prep_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Process dataset  errors:", result.stderr)

@app.local_entrypoint()
def main():
    from pathlib import Path
    import time
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    get_stats.remote()
    time.sleep(1)
    pull_hf_to_folder.remote() # run once to download the datasets
    time.sleep(1)

    # uncomment the next three lines to train the model
    # model_path = train_with_aim.remote(run_ts, yaml_path=f"train/yamls/finetune/{TRAIN_YAML}")
    # print(f"Model path: {model_path}")
    # model_path = Path(model_path).name
    # time.sleep(1)
    
    #view_model_checkpoints.remote()
    # time.sleep(1)
    # convert_model_to_hf.remote(model_path, yaml_path=f"train/yamls/finetune/{TRAIN_YAML}", upload_to_hf=False, is_peft=IS_PEFT)
    # time.sleep(1)
  
    # evaluate_model.remote(model_path)
    # time.sleep(1)

    # push_folder_to_hf.remote(Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/model_path) 
    # time.sleep(1)

    # generate_responses.remote(model_path)
