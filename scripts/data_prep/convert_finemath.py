from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import HfApi, login

hf_dataset = "LocalResearchGroup/finemath-4plus"
print(f"Enter you credentials to upload dataset {hf_dataset}")
login()


print(f"Downloading finemath-4plus")
dataset = load_dataset(
        path="HuggingFaceTB/finemath",
        name="finemath-4plus",
        split="train",
    )

# Split with a fixed seed
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

combined = DatasetDict({
    "train": split_dataset["train"],
    "valid": split_dataset["test"]
})

# push full dataset as is
print(f"Uploading finemath-4plus full train/val")
combined.push_to_hub(
    hf_dataset,
    config_name="full",
    private=True,
    max_shard_size="300MB")

# save to parquet locally
for split, dataset in combined.items():
    filename = f"dataset-{split}.parquet"
    if not Path().exists(filename).exists():
        print(f"Saving dataset-{split}.parquet")
        dataset.to_parquet(filename)
    else:
        print(f"dataset-{split}.parquet exist. Skipping...")

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

a = [1_000_000, 100_000, 10_000, 1000]
l = ["1M", "100k", "10k", "1k"]

for amount, label in zip(a, l):
    print(f"Creating ablation {label} train/val")
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
