## Quick Starts
Documentation on how to run the LLM-Foundry quick start (using SmolLM2-135M) on different platforms:

- [Modal](#modal)
- [Colab Pro](#colab-pro)
- [Paperspace Gradient Notebook](#paperspace-gradient-notebook)
- [Mac](#mac)


### Modal
[top](#quick-starts)

This assumes [you have modal installed and have logged in via command line](https://modal.com/docs/guide#:~:text=Create%20an%20account,modal%20setup). Go through the private wiki to create secrets on Modal. Make sure that [modal_script.py](https://github.com/LocalResearchGroup/llm-foundry/blob/main/scripts/modal/modal_script.py) is in the same folder as the Dockerfile below:

```
FROM mambaorg/micromamba:latest

USER root

# Install git and other dependencies
RUN apt-get update && apt-get install -y git nano curl wget && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone llm-foundry repo and set up environment
RUN git clone https://github.com/LocalResearchGroup/llm-foundry.git -b dataset-finemath /llm-foundry && \
    cd /llm-foundry && \
    micromamba create -n llm-foundry python=3.12 uv cuda -c nvidia/label/12.4.1 -c conda-forge && \
    export UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry && \
    micromamba run -n llm-foundry uv python pin 3.12 && \
    micromamba run -n llm-foundry uv sync --dev --extra gpu && \
    micromamba run -n llm-foundry uv sync --dev --extra gpu --extra flash --no-cache

ENV UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry
ENV CONDA_DEFAULT_ENV=llm-foundry
ENV PATH=/opt/conda/envs/llm-foundry/bin:$PATH

WORKDIR /llm-foundry

# Initialize conda in bash and activate environment by default
RUN echo "eval \"\$(micromamba shell hook --shell bash)\"" >> ~/.bashrc && \
    echo "micromamba activate llm-foundry" >> ~/.bashrc

# Open port to view Aim dashboard live from the container (optional) - Not related to aim remote upload server.
EXPOSE 43800

# Default shell with environment activated
CMD ["/bin/bash"]
```

This is the current Dockerfile we're using as we're working on testing things in the `dataset-finemath` branch. After you create the Dockerfile, your `modal` directory looks like this:
```
modal/
├── Dockerfile
├── modal_script.py
```

Make sure that you also have a local copy of the training YAML you want to use such as [smollm2-135m_lora_pretraining.yaml](https://github.com/LocalResearchGroup/llm-foundry/blob/main/scripts/train/yamls/pretrain/smollm2-135m_lora_pretraining.yaml).

In your modal_script.py make sure `pull_hf_to_folder.remote()` is uncommented. That function downloads all of our datasets to Modal. If you want to train, make sure the three lines related to training are uncommented as well. Here's the command to run the script on Modal (assume you are running the script in `modal` directory):

```
MODAL_GPU=A100-40GB TRAIN_YAML=../train/yamls/pretrain/smollm2-135m_lora_pretraining.yaml IS_PEFT=true OUTPUT_PRECISION=bf16 modal run modal_script.py 
```

Change `MODAL_GPU` and `TRAIN_YAML` as needed. `IS_PEFT=true` as the example YAML file is configured for LoRA. If all goes well you should see training logs wherever you're logging (W&B, AIM) and see printed logs in the terminal.

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
- `-p 43800:43800` run the aim server locally on port 43800 so you can watch progress live from your browser
```bash
docker run --rm -it --gpus device=0 --shm-size=8g -v /home/{your username}/llm_foundry/datasets/my-copy-c4:/root/my-copy-c4 -p 43800:43800 llm_foundry
```

Run the quickstart training command
```bash
cd /llm-foundry

# Set environment variables for AIM remote server upload - omit if only running locally
export AIM_CLIENT_REQUEST_HEADERS_SECRET='{"CF-Access-Client-Id": "xxxxxxxxxx", "CF-Access-Client-Secret": "xxxxxxxxxx"}'
export AIM_REMOTE_SERVER_URL_SECRET=https://{insert remote aim server url here}/upload

# OPTIONALLY: if you want to run the aim server in the background to view live progress in your browser - will print the url to your terminal w/ the public ip
echo "http://$(curl -s ifconfig.io):43800" && aim init -s && aim up --host 0.0.0.0 &

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
