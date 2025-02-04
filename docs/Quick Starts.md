## Quick Starts
Documentation on how to run the LLM-Foundry quick start (using SmolLM2-135M) on different platforms.


### Modal

TBD.

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
