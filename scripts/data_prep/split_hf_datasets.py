from argparse import ArgumentParser, Namespace

from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import HfApi, login
from pathlib import Path
import os

def save_to_parquet(combined: DatasetDict, out_ds_path: Path):
    data_files = dict()
    for split, dataset in combined.items():
        filename = out_ds_path /f"{split}.parquet"
        data_files[split] = filename
        if not Path(filename).exists():
            print(f"Saving {filename}")
            dataset.to_parquet(filename)
        else:
            print(f"{filename} already exist. Skipping...")
    return data_files

def create_size_ablation(dataset, total_rows):
    """Create a subset with a given percentage of the original data"""
    train_size = int(total_rows * 0.9)
    return {
        "train": dataset["train"].shuffle(42).select(range(train_size)),
        "test": dataset["test"].shuffle(42).select(range(total_rows - train_size)),
    }


def push_ablations(raw_datasets, ablations, hf_repo, config_name, private, shard_size):
    print(f"creating ablations from {len(raw_datasets['train'])}/{len(raw_datasets['test'])}")
    for label in ablations:
        match label[-1]:
            case "M":
                ds = create_size_ablation(raw_datasets, int(label[:-1]) * 1_000_000)
            case "k":
                ds = create_size_ablation(raw_datasets, int(label[:-1]) * 1_000)
            case _:
                ds = raw_datasets
    
        dsdict = DatasetDict(
            {
                "train": ds["train"],
                "test": ds["test"],
            }
        )
    
        print(f"Uploading ablation {label} train/val")
        
        dsdict.push_to_hub(hf_repo, config_name=label, private=private, max_shard_size=shard_size)

def pull_n_push(
    hf_ds_tgt,
    hf_ds_src,
    ds_name=None,
    after_pull=None,
    test_size: float = 0.1,
    seed: int = 42,
    saving2parquet=False,
    ablations = ("full", "1M", "100k", "10k", "1k"),
    private=True,
    shard_size: str = "300MB",
    purge_cache=False,
):
    banner = f"Loading dataset {hf_ds_src}/{'default' if ds_name is None else ds_name}"
    print("#"*len(banner))
    print(banner)
    print("#"*len(banner))
    dataset = load_dataset(path=hf_ds_src, name=ds_name, split="train")
    if after_pull is not None:
        dataset = after_pull(dataset)
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    dsd = DatasetDict({"train": dataset["train"], "test": dataset["test"]})

    if saving2parquet:
        print(f"Saving parquet to {hf_ds_tgt} train/test")
        out_ds_path = Path(hf_ds_tgt)
        out_ds_path.mkdir(parents=True, exist_ok=True)
        data_files = save_to_parquet(dsd, out_ds_path.absolute())
        # print(f"Loading parquet training/val from {str(out_ds_path)}\n\n\n")
        # dsd = load_dataset(str(out_ds_path))

    push_ablations(dsd, ablations, hf_ds_tgt, ds_name, private, shard_size)

    if purge_cache:
       dataset.cleanup_cache_files()

def filter_tulu(dataset):
    print(f"Original dataset rows {len(dataset)}")
    dataset = dataset.filter(lambda r: r["source"] is not None and "aya" not in r["source"])
    print(f"         current rows {len(dataset)}")
    return dataset

if __name__ == "__main__":
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("No Hugging Face token found. Please login.")
        login()
    pull_n_push("LocalResearchGroup/split-finemath", "HuggingFaceTB/finemath", "finemath-4plus")
    pull_n_push("LocalResearchGroup/split-tulu-3-sft-olmo-2-mixture", "allenai/tulu-3-sft-olmo-2-mixture", after_pull=filter_tulu, ablations=("full", "100k", "10k", "1k"))
    pull_n_push("LocalResearchGroup/split-NuminaMath-CoT", "AI-MO/NuminaMath-CoT", ablations=("full", "100k", "10k", "1k"))