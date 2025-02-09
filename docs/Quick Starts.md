## Quick Starts
Documentation on how to run the LLM-Foundry quick start (using SmolLM2-135M) on different platforms:

- [Modal](#modal)
- [Colab Pro](#colab-pro)
- [Paperspace Gradient Notebook](#paperspace-gradient-notebook)
- [Mac](#mac)


### Modal
[top](#quick-starts)

This assumes you have modal installed and have logged in via command line. Create the following file locally (we'll call it `quick-start.py`) and run it via CLI with `modal run quick-start.py`:

Note that the batch size arguments can be adjusted based on which GPU you use (L4, A10 or A100).

```python
from modal import Image, App

app = App("quick-start")

# Build image from local Dockerfile
image = Image.from_dockerfile("Dockerfile")

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

### Docker
[top](#quick-starts)

This assumes you have nvidia-docker installed. These are examples of command you may want to edit for your particular setup.

Create the dockerfile. You must be in the root directory of the repo
```bash
docker build -t llm_foundry .
```
Minimal run command. This will start interactively in bash `-it` and will be deleted when you exit `--rm`
```bash
sudo docker run --rm -it llm_foundry
```
Docker run command with additional options
- `--rm` delete the container when it exits
- `-it` interactive mode
- `--gpus device=0` use the first GPU only
- `--shm-size=8g` set the size of the shared memory - may be needed for higher # of cores w/ multiprocessing
- `-v /home/{your username}/llm_foundry/datasets/my-copy-c4:/root/my-copy-c4` mount a local directory to the container so, for example, you only have to download the dataset once for the quickstart

```bash
docker run --rm -it --gpus device=0 --shm-size=8g -v /home/{your username}/llm_foundry/datasets/my-copy-c4:/root/my-copy-c4 llm_foundry
```

Run the quickstart training command
```bash
cd /llm-foundry

# Set environment variables for AIM remote server upload - omit if only running locally
export AIM_CLIENT_REQUEST_HEADERS_SECRET='{"CF-Access-Client-Id": "xxxxxxxxxx", "CF-Access-Client-Secret": "xxxxxxxxxx"}'
export AIM_REMOTE_SERVER_URL_SECRET=https://{insert remote aim server url here}/upload

# Inside container, run training command
composer /llm-foundry/scripts/train/train.py \
  /llm-foundry/scripts/train/yamls/pretrain/smollm2-135m.yaml \
  variables.data_local=/root/my-copy-c4 \
  train_loader.dataset.split=train_small \
  eval_loader.dataset.split=val_small \
  max_duration=10ba \
  eval_interval=0 \
  save_folder=/root/smollm2-135m \
  device_eval_batch_size=4 \
  device_train_microbatch_size=4 \
  global_train_batch_size=4
```

Optionally - run the aim server locally by navigating to the directory containing the aim repo and run `aim up --host 0.0.0.0`
```bash
docker exec -it {container_name or container_id} bash
cd /llm-foundry/ # this folder for the quickstart or the folder containing the aim repo
aim up --host 0.0.0.0
```


### Colab Pro
[top](#quick-starts)

Google Colab does not come with conda installed so you have to run the following commands in a new terminal:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/miniconda3/bin/activate
conda init
```

Restart the session after running `conda init` and follow the installation instructions in the [main README](https://github.com/LocalResearchGroup/llm-foundry?tab=readme-ov-file#installation).

Create a file `train_smollm2-135m.sh` and upload it to the `/content` folder. This demo assumes that you have a `smollm2-135m.yaml` file:

<mark>Remove this line before merging PR: if the `docs` branch hasn't been merged to `main` upload `smollm2-135m.yaml` to `/content` and replace `train/yamls/pretrain/smollm2-135m.yaml` with `/content/smollm2-135m.yaml`</mark>

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

### Paperspace Gradient Notebook 
[top](#quick-starts)

We may want to change how we do this but for now I have created a Docker Hub registry `vishalbakshi/llm-foundry-jupyter:latest` in [this repo](https://github.com/vishalbakshi/llm-foundry-dockerfile). I believe there is a 1 free Docker Hub registry per account.

In paperspace:

1. Create a notebook.
2. Select "Start from Scratch" template.
3. Paste `vishalbakshi/llm-foundry-jupyter:latest` into the Container name and leave all other fields as is.
4. Click Start Notebook (for me, it fails the first time so I have to repeat steps 2 and 3).

Open a terminal and type:

```
cd /app/llm-foundry
```

<mark>Remove this line before merging PR: if the `docs` branch hasn't been merged to `main` yet use `git checkout docs`</mark>

Save the script from the Colab Pro section and upload it to your Paperspace notebook (it will live in the `/notebooks` folder). Run the following line (replace `train_smollm2-135m.sh` with whatever you named your `.sh` file):

```
../../notebooks/train_smollm2-135m.sh
```
You should see the output logs as described in the Modal section.

### Mac
[top](#quick-starts)

The following command is confirmed to work with M1:

```
composer train/train.py \
  train/yamls/pretrain/mpt-125m.yaml \
  variables.data_local=my-copy-c4 \
  train_loader.dataset.split=train_small \
  eval_loader.dataset.split=val_small \
  max_duration=10ba \
  eval_interval=0 \
  save_folder=mpt-125m \
  model.attn_config.attn_impl=torch \
  model.loss_fn=torch_crossentropy \
  precision=FP32 \
  train_loader.num_workers=0 \
  eval_loader.num_workers=0 \
  global_train_batch_size=32 \
  device_train_microbatch_size=8 \
  train_loader.prefetch_factor=null \
  eval_loader.prefetch_factor=null \
  train_loader.persistent_workers=false \
  eval_loader.persistent_workers=false
```

We haven't yet been able to get `composer eval/eval.py` working. It hits `torch.cuda.current_device()` in `composer/distributed/dist_strategy.py` and then errors because torch was compiled without CUDA. 
