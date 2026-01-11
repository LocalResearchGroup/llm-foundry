
import os
import modal
import sys
from modal import Image, App, Secret, Volume

import pathlib, datetime

PYTHON_PATH = "/opt/conda/envs/llm-foundry/bin/python"

# command line arguments
TRAINING_GPU = os.environ.get("MODAL_GPU", "L4") 
TRAIN_YAML = os.environ.get("TRAIN_YAML", "")
from modal import Image, App, Volume, Secret
# Build image from local Dockerfile
image = Image.from_dockerfile("Dockerfile", gpu='L4')
image = image.add_local_file(TRAIN_YAML, f"/llm-foundry/scripts/train/yamls/finetune/{TRAIN_YAML}")
import os
import pathlib

TRAINING_GPU = os.environ.get("MODAL_GPU", "L4") 
app = App("ds-stats")


DATASETS_VOLUME = Volume.from_name("lrg-datasets", create_if_missing=True)
DATASETS_VOLUME_MOUNT_PATH = pathlib.Path("/datasets")

DATASET_PATHS = {
    "tulu": "/datasets/cleaned/tulu-tokens",
    "numina": "/datasets/cleaned/numina-tokens",
    "glaive": "/datasets/cleaned/glaive-tokens",
    "finemath": "/datasets/cleaned/finemath-tokens",
    "pythonedu": "/datasets/cleaned/pythonedu-tokens",
}

@app.function(gpu=TRAINING_GPU, image=image, timeout=3*3600, secrets=[Secret.from_name("LRG")],
              volumes={DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def pull_hf_to_folder():
    print("---------------------------------------------------------\n"*77)
    import subprocess
    import os



    print(f"Working directory 1: {os.getcwd()}")
    print(f"Working tree root 1: {os.system('ls /root/')}")
    print(f"Working tree / 1: {os.system('ls /')}")
    # Change to llm-foundry/scripts directory at the start
    os.chdir("/llm-foundry/scripts")
    print(f"Working directory: {os.getcwd()}")

    # Step 1: pull all tokens
    print(f"Downloading repos to {DATASETS_VOLUME_MOUNT_PATH}/")
    data_prep_cmd = [
        PYTHON_PATH,  # Use the correct Python interpreter
        "data_prep/download_tokens.py",
        "--decontaminated",
        "--out", f"{DATASETS_VOLUME_MOUNT_PATH}/cleaned/",
    ]
    result = subprocess.run(data_prep_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Download data errors:", result.stderr)

    DATASETS_VOLUME.commit()


SPLITS = ["train", "test"]
MAX_SEQ_LEN = 8192

@app.function(image=image, timeout=3*3600, volumes={DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME})
def compute_stats(dataset_name: str):
    os.system("uv pip list | grep num")
    print("---###+++ ! +++###---\n"*3)
    os.system("uv pip show numpy")
    print("---###+++ activate and check!!! ###---\n"*27)
    os.system("ls ~")
    os.system("which python")
    os.system("whoami")
    os.system("echo -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-")
    os.system("uv pip list | grep numpy")
    os.system("uv pip check")
    os.system("uv pip show numpy")
    os.system("uv pip show mosaicml-streaming")
    os.system("echo -0-0-0-0-0-0-0-0- 888888888888 -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-")
    print("\n"*7)
    print("sys")
    import sys
    print(sys.executable)
    print(sys.path)
    print("\n"*7)
    print("---####+++###-###+++###-###*******##+++###-###+++###-##+++###---\n"*13)
    print("\n"*7)
    os.system("uv pip show numpy")
    os.system("echo -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-")
    import numpy as np
    print("---###+++###-###+++###-##+++###---\n"*27)
    all_results = []
    from streaming import StreamingDataset
    from transformers import AutoTokenizer
    from tqdm import tqdm
    print("========================= check again!!!! ======================\n\n"*7)
    os.system("which python")
    os.system("whoami")
    os.system("echo -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-")
    os.system("uv pip list | grep numpy")
    os.system("uv pip check")
    os.system("uv pip show numpy")
    os.system("uv pip show mosaicml-streaming")
    os.system("echo -0-0-0-0-0-0-0-0- 888888888888 -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-")

    import numpy as np
    print("                      OK!!!!!\n"*7)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    eos_id = tokenizer.eos_token_id
    special_tokens = {
        "eos": eos_id,
        "im_start": tokenizer.encode("<|im_start|>", add_special_tokens=False)[0],
        "im_end": tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
    }

    base_path = DATASET_PATHS.get(dataset_name)
    if not base_path:
        print(f"Unknown dataset: {dataset_name}")
        return []

    results = []
    for split in reversed(SPLITS):
        path = f"{base_path}/{split}"
        if not os.path.exists(path):
            continue

        try:
            print(f"\nopening folder {path=}\n\n\n")
            ds = StreamingDataset(remote=None, local=path, batch_size=1, shuffle=False)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        lengths = []
        token_counts = {k: 0 for k in special_tokens}
        total_tokens = 0

        for item in tqdm(ds, desc=f"{dataset_name}/{split}"):
            if "tokens" in item:
                toks = item["tokens"]
            elif "turns" in item:
                toks = item["turns"][0]["input_ids"]
            else:
                continue

            toks = np.array(toks)
            seq_len = len(toks)
            lengths.append(seq_len)
            total_tokens += seq_len

            for name, tok_id in special_tokens.items():
                token_counts[name] += int(np.sum(toks == tok_id))

        if not lengths:
            continue

        lengths = np.array(lengths)
        stats = {
            "dataset": dataset_name,
            "split": split,
            "total_tokens": total_tokens,
            "num_sequences": len(lengths),
            "avg_length": round(np.mean(lengths), 2),
            "median_length": int(np.median(lengths)),
            "std_length": round(np.std(lengths), 2),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "seqs_over_max": int(np.sum(lengths > MAX_SEQ_LEN)),
            "pct_over_max": round(100 * np.sum(lengths > MAX_SEQ_LEN) / len(lengths), 2),
            "p25": int(np.percentile(lengths, 25)),
            "p50": int(np.percentile(lengths, 50)),
            "p75": int(np.percentile(lengths, 75)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99)),
            "eos_count": token_counts["eos"],
            "im_start_count": token_counts["im_start"],
            "im_end_count": token_counts["im_end"],
        }
        results.append(stats)
        print(f"Done: {dataset_name}/{split} - {len(lengths)} seqs, {total_tokens:,} tokens")

    return results

@app.function(image=image, timeout=3*3600, volumes={DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME})
def save_results(all_results):
    print(f"\n\n\nsave_results")
    print(len(all_results))
    print(type(all_results))
    print("\n"*11)
    cols = ["dataset", "split", "total_tokens", "num_sequences", "avg_length", "median_length", 
            "std_length", "min_length", "max_length", "seqs_over_max", "pct_over_max", 
            "p25", "p50", "p75", "p95", "p99", "eos_count", "im_start_count", "im_end_count"]

    lines = []
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["------" for _ in cols]) + "|"
    lines.append(header)
    lines.append(sep)

    for r in all_results:
        row = "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        lines.append(row)

    lines.append("\n## Special Token Check\n")
    for r in all_results:
        lines.append(f"**{r['dataset']}/{r['split']}:**")
        lines.append(f"- EOS tokens: {r['eos_count']:,} ({r['eos_count']/r['num_sequences']:.2f} per seq)")
        lines.append(f"- <|im_start|>: {r['im_start_count']:,} ({r['im_start_count']/r['num_sequences']:.2f} per seq)")
        lines.append(f"- <|im_end|>: {r['im_end_count']:,} ({r['im_end_count']/r['num_sequences']:.2f} per seq)")
        lines.append("")

   
    output_path =  f"{DATASETS_VOLUME_MOUNT_PATH}/ds_stats.md"
    print(f"writin results to {output_path}")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    DATASETS_VOLUME.commit()
    print(f"Saved 2 to {output_path}")
    return "\n".join(lines)

@app.local_entrypoint()
def main():
    print("---+++---\n"*27)
    if True:
        pull_hf_to_folder.remote()
        return

    print(f"{DATASET_PATHS}\n\n\n\n\n")
    all_results = []
    for dataset_name in DATASET_PATHS.keys():
        print(f"processing {dataset_name}")
        results = compute_stats.remote(dataset_name)
        all_results.extend(results)

    print("--- results??? ---\n"*27)
    if not all_results:
        print("No results found")
        return


    print("--- ####### ---\n"*13)
    output = save_results.remote(all_results)
    print("\n\n                              this are the ouputs\n\n")
    print(output)
    print("\n\nEOT\n\n")

