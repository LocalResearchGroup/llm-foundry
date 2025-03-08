from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import HfApi, login
from pathlib import Path

hf_dataset = "LocalResearchGroup/tulu-3-train-val"
print(f"Enter you credentials to upload dataset {hf_dataset}")
login()


print(f"Downloading allenai/tulu-3-sft-olmo-2-mixture")
dataset = load_dataset(
        path="allenai/tulu-3-sft-olmo-2-mixture",
        split="train",
    )

print(f"Original dataset rows {len(dataset)}")
dataset = dataset.filter(lambda r: r["source"] is not None and "aya" not in r["source"])
print(f"Without aya rows {len(dataset)}")

# Split with a fixed seed
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

combined = DatasetDict({
    "train": split_dataset["train"],
    "valid": split_dataset["test"]
})
print(f"train/splits: {len(combined['train'])}/{len(combined['valid'])} rows")

# push full dataset as is
print(f"Uploading {hf_dataset} full train/val")
combined.push_to_hub(
    hf_dataset,
    config_name="full",
    private=True,
    max_shard_size="300MB")

# save to parquet locally
for split, dataset in combined.items():
    filename = f"dataset-{split}.parquet"
    if not Path(filename).exists():
        print(f"Saving dataset-{split}.parquet with {len(dataset)} rows")
        dataset.to_parquet(filename)
    else:
        print(f"dataset-{split}.parquet exist. Skipping...")

###############################
# load from local
data_files = {
    "train": "dataset-train.parquet",
    "valid": "dataset-valid.parquet"
}

print("Loading parquet training/val")
raw_datasets = load_dataset("parquet", data_dir=".", data_files=data_files)

def create_size_ablation(dataset, total_rows):
    """Create a subset with a given percentage of the original data"""
    train_size = int(total_rows * .9)
    return {
        "train": dataset["train"].shuffle(42).select(range(train_size)),
        "valid": dataset["valid"].shuffle(42).select(range(total_rows - train_size)),
    }

a = [100_000, 10_000, 1000]
l = ["100k", "10k", "1k"]

for amount, label in zip(a, l):
    print(f"Creating ablation {label} from original train/val sets")
    ds =create_size_ablation(raw_datasets, amount)

    dsdict = DatasetDict({
        "train": ds["train"],
        "valid": ds["valid"],
    })

    print(f"Uploading ablation {label} train/val")
    dsdict.push_to_hub(
        hf_dataset,
        config_name=label,
        private=False
    )
