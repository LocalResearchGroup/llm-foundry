## Quick Starts
Documentation on how to run the LLM-Foundry quick start (using SmolLM2-135M) on different platforms.


### Modal

This assumes you have modal installed and have logged in via command line. Create the following file locally (we'll call it `quick-start.py`) and run it via CLI with `modal run quick-start.py`:

<mark>Remove this line before opening PR: if the `docs` branch hasn't been merged to `main` yet use `"git clone -b docs https://github.com/LocalResearchGroup/llm-foundry.git && "`</mark>

```python
from modal import Image, App

def setup_image():
    return (
        Image.micromamba()
        .apt_install(["git"])
        .run_commands(
            "git clone https://github.com/LocalResearchGroup/llm-foundry.git && "
            "cd llm-foundry && "
            "micromamba create -n llm-foundry python=3.12 cuda uv -c nvidia/label/12.4.1 -c conda-forge -y && "
            "export UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry && "
            "micromamba run -n llm-foundry uv sync --extra dev --extra gpu && "
            "micromamba run -n llm-foundry uv sync --extra flash && "
            "cd /llm-foundry && micromamba run -n llm-foundry pip install -e ."
        )
        .env({"CONDA_DEFAULT_ENV": "llm-foundry", 
              "PATH": "/opt/conda/envs/llm-foundry/bin:$PATH"})
    )

app = App("quick-start-smollm2-135-m")
image = setup_image()

@app.function(gpu="l4", image=image, timeout=3600)
def run_quickstart():
    import subprocess
    import os
    
    # Ensure we're using the right Python
    python_path = "/opt/conda/envs/llm-foundry/bin/python"
    
    # Use the correct Python interpreter for imports
    import_check = subprocess.run(
        [python_path, "-c", "import flash_attn; print(flash_attn.__version__)"],
        capture_output=True,
        text=True
    )
    print(f"Flash Attention version: {import_check.stdout}")

    # Change to llm-foundry/scripts directory at the start
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Convert C4 dataset
    print("Converting C4 dataset...")
    data_prep_cmd = [
        python_path,  # Use the correct Python interpreter
        "data_prep/convert_dataset_hf.py",
        "--dataset", "allenai/c4",
        "--data_subset", "en",
        "--out_root", "/root/my-copy-c4",
        "--splits", "train_small", "val_small",
        "--concat_tokens", "2048",
        "--tokenizer", "HuggingFaceTB/SmolLM2-135M",
        "--eos_text", "<|endoftext|>"
    ]
    result = subprocess.run(data_prep_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Data prep errors:", result.stderr)
    
    # Step 2: Train the model
    print("\nTraining model...")
    train_cmd = [
        "composer",
        "train/train.py",
        "train/yamls/pretrain/smollm2-135m.yaml",  # Updated YAML path
        "variables.data_local=/root/my-copy-c4",
        "train_loader.dataset.split=train_small",
        "eval_loader.dataset.split=val_small",
        "max_duration=10ba",
        "eval_interval=0",
        "save_folder=/root/smollm2-135m",  # Updated model name
        "device_eval_batch_size=4",  # Added batch size settings
        "device_train_microbatch_size=4",
        "global_train_batch_size=4"
    ]
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Training errors:", result.stderr)
    if result.returncode != 0:
        raise Exception(f"Training failed with exit code {result.returncode}\nStderr: {result.stderr}")
    
    # Step 3: Convert model to HuggingFace format
    print("\nConverting model to HuggingFace format...")
    convert_cmd = [
        python_path, "inference/convert_composer_to_hf.py",
        "--composer_path", "/root/smollm2-135m/ep0-ba10-rank0.pt",  # Updated path
        "--hf_output_path", "/root/smollm2-135m-hf",  # Updated path
        "--output_precision", "bf16"
    ]
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Conversion errors:", result.stderr)
    
    # Step 4: Evaluate the model
    print("\nEvaluating model...")
    eval_cmd = [
        "composer",
        "eval/eval.py",
        "eval/yamls/hf_eval.yaml",
        "icl_tasks=eval/yamls/copa.yaml",
        "variables.model_name_or_path=/root/smollm2-135m-hf"  # Updated path
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Evaluation errors:", result.stderr)
    
    # Step 5: Generate test responses
    print("\nGenerating test responses...")
    generate_cmd = [
        python_path, "inference/hf_generate.py",
        "--name_or_path", "/root/smollm2-135m-hf",  # Updated path
        "--max_new_tokens", "256",
        "--prompts",
        "The answer to life, the universe, and happiness is",
        "Here's a quick recipe for baking chocolate chip cookies: Start by"
    ]
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Generation errors:", result.stderr)
    
    return "Quickstart completed!"

@app.local_entrypoint()
def main():
    print(run_quickstart.remote())

```

If all goes well you should see something like the following during training:

```
[batch=1/10]:
         Train time/epoch: 0
         Train time/batch: 0
         Train time/sample: 0
         Train time/batch_in_epoch: 0
         Train time/sample_in_epoch: 0
         Train time/token: 0

...
```

something like the following during evaluation:

```
Printing complete results for all models
| Category              | Benchmark   | Subtask   |   Accuracy | Number few shot   | Model           |
|:----------------------|:------------|:----------|-----------:|:------------------|:----------------|
| commonsense_reasoning | copa        |           |       0.49 | 0-shot            | smollm2-135m-hf |
```

and something like the following during generation:

```
The answer to life, the universe, and happiness is**starting Fest Discover shelf reproduce...
####################################################################################################
Here's a quick recipe for baking chocolate chip cookies: Start by quinoa Quantum availabilityromb...
```

### Colab Pro

Google Colab does not come with conda installed so you have to run the following commands in a new terminal:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate
conda init
```

Restart the session after running `conda init`. 

Create a file `train_smollm2-135m.sh` and upload it to the `/content` folder. This demo assumes that you have a `smollm2-135m.yaml` file:

```bash
cd scripts

# Convert C4 dataset to StreamingDataset format
python data_prep/convert_dataset_hf.py \
  --dataset allenai/c4 --data_subset en \
  --out_root my-copy-c4 --splits train_small val_small \
  --concat_tokens 2048 --tokenizer HuggingFaceTB/SmolLM2-135M --eos_text '<|endoftext|>'

# Train SmolLM2-135M model for 10 batches
composer train/train.py \
  train/yamls/pretrain/smollm2-135m.yaml \
  variables.data_local=my-copy-c4 \
  train_loader.dataset.split=train_small \
  eval_loader.dataset.split=val_small \
  max_duration=10ba \
  eval_interval=0 \
  save_folder=smollm2-135m \
  device_eval_batch_size=4 \
  device_train_microbatch_size=4 \
  global_train_batch_size=4

# Convert the model to HuggingFace format
python inference/convert_composer_to_hf.py \
  --composer_path smollm2-135m/ep0-ba10-rank0.pt \
  --hf_output_path smollm2-135m-hf \
  --output_precision bf16 \
  # --hf_repo_for_upload user-org/repo-name

# Evaluate the model on a subset of tasks
composer eval/eval.py \
  eval/yamls/hf_eval.yaml \
  icl_tasks=eval/yamls/copa.yaml \
  variables.model_name_or_path=smollm2-135m-hf

# Generate responses to prompts
python inference/hf_generate.py \
  --name_or_path smollm2-135m-hf \
  --max_new_tokens 256 \
  --prompts \
    "The answer to life, the universe, and happiness is" \
    "Here's a quick recipe for baking chocolate chip cookies: Start by"
```

Note that the batch size arguments can be adjusted based on which GPU you use (L4 or A100).

### Paperspace Gradient Notebook 

TBD.

### Mac

TBD
