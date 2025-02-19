#TODO:
# - Probably want to put local aim repo in the folder with the model checkpoints
# - Fix duplicate container issue
# - Add function to publish model to hf hub. Add `huggingface_hub` to project dependencies and thus to the project container. 
#   - https://huggingface.co/docs/huggingface_hub/installation 
#   - https://huggingface.co/docs/huggingface_hub/guides/upload



import modal
from modal import Image, App, Secret, Volume

import pathlib, datetime

PYTHON_PATH = "/opt/conda/envs/llm-foundry/bin/python"
TRAINING_GPU = "l4" # "a10g" "h100" # "l4"
BATCH_SIZE = 4 # 20 for h100 4 for l4
DATASET_BASE_PATH = "/datasets"
DATASETS_VOLUME = Volume.from_name("lrg-datasets", create_if_missing=True)
DATASETS_VOLUME_MOUNT_PATH = pathlib.Path("/datasets")
MODEL_CHECKPOINT_VOLUME = Volume.from_name("lrg-model-checkpoints", create_if_missing=True)
MODEL_CHECKPOINT_VOLUME_MOUNT_PATH = pathlib.Path("/model-checkpoints")

app = App("quick-start")

# Build image from local Dockerfile
image = Image.from_dockerfile("Dockerfile")

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
             concurrency_limit=1)
def get_stats():
    import subprocess
    
    # Use the correct Python interpreter for imports
    import_check = subprocess.run(
        [PYTHON_PATH, "-c", "import flash_attn; print(flash_attn.__version__)"],
        capture_output=True,
        text=True
    )
    print(f"Flash Attention version: {import_check.stdout}")

    # Run nvidia-smi to check GPU status
    nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print("NVIDIA-SMI Output:")
    print(nvidia_smi.stdout)
    if nvidia_smi.stderr:
        print("NVIDIA-SMI Errors:", nvidia_smi.stderr)


@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")], 
              volumes={DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              concurrency_limit=1)
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
        "--eos_text", "<|endoftext|>"
    ]
    result = subprocess.run(data_prep_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Data prep errors:", result.stderr)
    
    DATASETS_VOLUME.commit()

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")], 
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              concurrency_limit=1)
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


# @app.function(gpu=TRAINING_GPU, image=image, timeout=6*3600, secrets=[Secret.from_name("LRG")], 
#               volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
#                        DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME})
def train_model(run_ts: str):
    import subprocess
    import os
    
    # Change to llm-foundry/scripts directory at the start
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 2: Train the model
    print("\nTraining model...")
    run_folder = f"{MODEL_CHECKPOINT_VOLUME_MOUNT_PATH}/smollm2-135m-{run_ts}"
    save_folder = f"{run_folder}/native_checkpoints"
    
    train_cmd = [
        "composer",
        "train/train.py",
        "train/yamls/pretrain/smollm2-135m.yaml",  # Updated YAML path
        "loggers.aim.experiment_name=quickstart_test_smollm2_135m_modal",
        f"variables.data_local={DATASETS_VOLUME_MOUNT_PATH}/c4_small",
        "train_loader.dataset.split=train_small",
        "eval_loader.dataset.split=val_small",
        "max_duration=100ba",
        "eval_interval=100ba",
        f"save_folder={save_folder}",  # Updated model name
        f"device_eval_batch_size={BATCH_SIZE}",  # Added batch size settings # 20 for h100 4 for l4
        f"device_train_microbatch_size={BATCH_SIZE}",
        f"global_train_batch_size={BATCH_SIZE}"
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

def run_aim_server():
    import subprocess
    import os
    
    os.chdir("/llm-foundry/scripts")
    print("Initializing Aim...")
    subprocess.run(["aim", "init"], check=True)
    
    process = subprocess.Popen(
        ["aim", "up", "--host", "0.0.0.0", "--port", "43800"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return process


@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                      DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              concurrency_limit=1)
def train_with_aim(run_ts: str):
    import subprocess, time

    with modal.forward(43800) as tunnel:
        print(f"\nAim server available at: {tunnel.url}")
        model_path = None
        aim_task = run_aim_server()
        time.sleep(5)
    
        try:
            model_path = train_model(run_ts)

        finally:
            aim_task.terminate()
            try:
                aim_task.wait(timeout=5)
            except subprocess.TimeoutExpired:
                aim_task.kill()
    
    return model_path

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              concurrency_limit=1)
def convert_model_to_hf(checkpoint_path: str):
    """
    Convert a model checkpoint to a HuggingFace format.
    """
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
        "--output_precision", "bf16"
    ]
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Conversion errors:", result.stderr)
    MODEL_CHECKPOINT_VOLUME.commit()
    print("Conversion complete!")

@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              concurrency_limit=1)
def push_to_hf(checkpoint_path: str):
    from huggingface_hub import create_repo, upload_folder
    from pathlib import Path
    
    model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
    model_name = model_path.name
    repo_id = f"LocalResearchGroup/{model_name}"
    
    print(f"Creating repository: {repo_id}")
    create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

    upload_folder(folder_path=model_path, repo_id=repo_id)
    
    print(f"Folder uploaded to: https://huggingface.co/{repo_id}")
  
@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              concurrency_limit=1)
def evaluate_model(checkpoint_path: str):
    import subprocess, os
    from pathlib import Path

    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path
    save_path = model_path.parent/"evals"  # Create evals subfolder path
    
    print("\nEvaluating model...")
    eval_cmd = [
        "composer",
        "eval/eval.py",
        "eval/yamls/hf_eval.yaml",
        "icl_tasks=eval/yamls/copa.yaml",
        f"variables.model_name_or_path={model_path}",
        f"results_path={save_path}"  # Add results_path parameter
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Evaluation errors:", result.stderr)
    
    MODEL_CHECKPOINT_VOLUME.commit()  # Commit the new eval results
    print("Evaluation complete!")


@app.function(gpu=TRAINING_GPU, image=image, timeout=3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME},
              concurrency_limit=1)
def generate_responses(checkpoint_path: str, prompts: list[str]|str|None=None):
    import subprocess, os
    from pathlib import Path

    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    model_path = Path(MODEL_CHECKPOINT_VOLUME_MOUNT_PATH)/checkpoint_path

    if prompts is None:
        prompts = [
            "The answer to life, the universe, and happiness is",
            "Here's a quick recipe for baking chocolate chip cookies: Start by"
        ]
    elif isinstance(prompts, str):
        prompts = [prompts]
    

    print("\nGenerating test responses...")
    generate_cmd = [
        PYTHON_PATH, "inference/hf_generate.py",
        "--name_or_path", model_path,
        "--max_new_tokens", "256",
        "--prompts",
        *prompts
    ]
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Generation errors:", result.stderr)
    print("Generation complete!")


@app.local_entrypoint()
def main():
    from pathlib import Path
    import time
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    get_stats.remote()
    time.sleep(1)
    # convert_c4_small_dataset.remote() # Only run once

    model_path = train_with_aim.remote(run_ts)
    print(f"Model path: {model_path}")
    model_path = Path(model_path).name
    time.sleep(1)
    
    view_model_checkpoints.remote()
    time.sleep(1)
    convert_model_to_hf.remote(model_path)
    time.sleep(1)

    push_to_hf.remote(model_path)
    time.sleep(1)
  
    evaluate_model.remote(model_path)
    time.sleep(1)
    generate_responses.remote(model_path)
