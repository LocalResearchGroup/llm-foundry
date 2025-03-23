from argparse import ArgumentParser, Namespace

from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import HfApi, login
from pathlib import Path
import os

def save_to_parquet(combined: DatasetDict, out_ds_path: Path):
    data_files = {}
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
            },
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
    # filter out messages of lenght = 2 user+assistant
    # FIXME: extra checks finding [None] * 512 batch?
    dataset = dataset.filter(lambda r: r["source"] is not None and "aya" not in r["source"] and len(r["messages"]) == 2 and r["messages"] is not None and r["messages"][0] is not None and r["messages"][1] is not None)
    print("tulu", dataset.features)
    dataset = dataset.remove_columns(["source", "dataset"])
    print("new tulu features: ", dataset.features)
    print(f"         current rows {len(dataset)}")
    return dataset

def process_numina(dataset):
    print("numina", dataset.features)
    # remove column that on batch of 512 only has 2 rows which breaks pytorch collate!
    dataset = dataset.remove_columns("messages")
    print("new numina features", dataset.features)
    return dataset

def upload_token_folder(local_path, target_repo):
    # WIP: remove constants
    api = HfApi()
    p = Path(".")
    api.upload_folder(
        folder_path=local_path,
        repo_id=target_repo,
        repo_type="dataset",
        # path_in_repo="",
        # commit_message="",
    )


def create_upload():
    # import configurations to tokenize new dataset splits
    import tokenize_split
    from llmfoundry.command_utils import convert_dataset_hf_from_args, DatasetConstants, DataSplitConstants, add_dataset_config, CONSTS

    configs = [
        {
            "target": "tyoc213/split-tulu-3-sft-olmo-2-mixture",
            "ablations": ["full", "100k", "10k", "1k"],
        },
        {
            "target": "tyoc213/split-NuminaMath-CoT",
            "ablations": ["full", "100k", "10k", "1k"],
        },
        # {
        #     "target": "tyoc213/split-finemath",
        #     "ablations": ["full", "1M", "100k", "10k", "1k"],
        # },
    ]

    for c in configs:
        folder = c["target"].split("/")[1]
        for ablation in c["ablations"]:
            args = Namespace(
                dataset=c["target"],
                data_subset=ablation,
                splits=['train', 'test'],
                out_root=f'tokenized-{folder}-{ablation}',
                compression="zstd",
                concat_tokens=None,
                tokenizer='HuggingFaceTB/SmolLM2-135M',
                tokenizer_kwargs=None,
                bos_text=None,
                eos_text='<|endoftext|>',
                no_wrap=False,
                num_workers=None,
            )

            convert_dataset_hf_from_args(
                dataset=args.dataset,
                data_subset=args.data_subset,
                splits=args.splits,
                out_root=args.out_root,
                compression=args.compression,
                concat_tokens=args.concat_tokens,
                tokenizer=args.tokenizer,
                tokenizer_kwargs=args.tokenizer_kwargs,
                bos_text=args.bos_text,
                eos_text=args.eos_text,
                no_wrap=args.no_wrap,
                num_workers=args.num_workers,
            )

    # upload all tokenized folders to corresponding repo/folder
    for c in configs:
        for ablation in c["ablations"]:
            local_path = Path(".") / f"{c['target']}" / f"{ablation}"
            target_repo = c["target"]
            # upload_token_folder(local_path, target_repo)
if __name__ == "__main__":
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("No Hugging Face token found. Please login.")
        login()
    REMOTE_REPO = "LocalResearchGroup"
    pull_n_push(f"{REMOTE_REPO}/split-tulu-3-sft-olmo-2-mixture", "allenai/tulu-3-sft-olmo-2-mixture", after_pull=filter_tulu, ablations=("full", "100k", "10k", "1k"))
    pull_n_push(f"{REMOTE_REPO}/split-NuminaMath-CoT", "AI-MO/NuminaMath-CoT", after_pull=process_numina, ablations=("full", "100k", "10k", "1k"))
    pull_n_push(f"{REMOTE_REPO}/split-finemath", "HuggingFaceTB/finemath", "finemath-4plus", ablations=("100k", "10k", "1k"))

    if False:
        create_upload()
