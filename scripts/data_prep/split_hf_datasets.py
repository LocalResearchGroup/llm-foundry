from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import HfApi, login
from pathlib import Path
import os


def split_dataset(
    inp_ds_path: str,
    inp_ds_name: str | None = None,
    out_ds_path: str | None = None,
    test_size: float = 0.1,
    seed: int = 42,
    shard_size: str = "300MB",
):
    """Split a dataset into train and test sets and save to disk"""
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("No Hugging Face token found. Please login.")
        login()
    
    print(f"Loading dataset {inp_ds_path}/{inp_ds_name}")
    dataset = load_dataset(path=inp_ds_path, name=inp_ds_name, split="train")

    print(f"Splitting dataset")
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    print(f"Saving dataset to {out_ds_path}")
    split_dataset.save_to_disk(out_ds_path)

    combined = DatasetDict({"train": split_dataset["train"], "valid": split_dataset["test"]})

    print(f"Uploading dataset to {out_ds_path}")
    combined.push_to_hub(out_ds_path, config_name="full", private=True, max_shard_size=shard_size)

    # save splits locally
    for split, dataset in combined.items():
        filename = f"dataset-{split}.parquet"
        if not Path(filename).exists():
            print(f"Saving dataset-{split}.parquet")
            dataset.to_parquet(filename)
        else:
            print(f"dataset-{split}.parquet exist. Skipping...")

    # load from local
    data_files = {"train": "dataset-train.parquet", "valid": "dataset-valid.parquet"}

    print("Loading parquet training/val")
    raw_datasets = load_dataset("parquet", data_dir=".", data_files=data_files)


def create_size_ablation(dataset, total_rows):
    """Create a subset with a given percentage of the original data"""
    train_size = int(total_rows * 0.9)
    return {
        "train": dataset["train"].shuffle(42).select(range(train_size)),
        "valid": dataset["valid"].shuffle(42).select(range(total_rows - train_size)),
    }


a = [1_000_000, 100_000, 10_000, 1000]
l = ["1M", "100k", "10k", "1k"]

for amount, label in zip(a, l):
    print(f"Creating ablation {label} train/val")
    ds = create_size_ablation(raw_datasets, amount)

    dsdict = DatasetDict(
        {
            "train": ds["train"],
            "valid": ds["valid"],
        }
    )

    print(f"Uploading ablation {label} train/val")
    dsdict.push_to_hub(hf_dataset, config_name=label, private=False)
