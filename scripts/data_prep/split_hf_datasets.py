from argparse import ArgumentParser, Namespace, BooleanOptionalAction

from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import HfApi, login
from pathlib import Path

from convert_finetuning_dataset import convert_finetuning_dataset_from_args
import os

import dataset_constants_split_config
from llmfoundry.command_utils import convert_dataset_hf_from_args


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
    
        print(f"\nUploading ablation {label} train/val")
        
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
    print(f"path={hf_ds_src=}, name={ds_name=}, split=train")
    print("#"*len(banner))

    dataset = load_dataset(path=hf_ds_src, name=ds_name, split="train")
    if after_pull is not None:
        dataset = after_pull(dataset)
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    dsd = DatasetDict({"train": dataset["train"], "test": dataset["test"]})

    if saving2parquet:
        b = f"Saving parquet to {hf_ds_tgt} train/test"
        print("=" * len(b))
        print(b)
        print("=" * len(b))
        out_ds_path = Path(hf_ds_tgt)
        out_ds_path.mkdir(parents=True, exist_ok=True)
        data_files = save_to_parquet(dsd, out_ds_path.absolute())

    push_ablations(dsd, ablations, hf_ds_tgt, ds_name, private, shard_size)

    if purge_cache:
       dataset.cleanup_cache_files()

def filter_tulu(dataset):
    print(f"Original dataset rows {len(dataset)}")
    dataset = dataset.filter(lambda r: r["source"] is not None and "aya" not in r["source"] and len(r["messages"]) == 2)
    print("tulu", dataset.features)
    dataset = dataset.remove_columns(["source", "dataset"])
    def extract_qa(messages):
        user_question = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
        assistant_response = next((msg["content"] for msg in messages if msg["role"] == "assistant"), None)
        return {"prompt": user_question, "response": assistant_response}

    # Apply function to dataset
    dataset = dataset.map(lambda example: extract_qa(example["messages"]))
    dataset = dataset.remove_columns(["messages"])
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
    print(f"upload_token_folder({str(local_path.relative_to('.'))}, {target_repo})")
    api = HfApi()
    r = api.upload_folder(
        folder_path=local_path,
        repo_id=target_repo,
        repo_type="dataset",
        path_in_repo=str(local_path.relative_to(".")),
    )
    print(f"token uploaded result: {r}")


def create_pretraining_tokens(args, datasets, tokenizer="HuggingFaceTB/SmolLM2-135M"):
    # import configurations to tokenize new dataset splits
    
    for s in args.source:
        
        d = datasets[s]
        folder = d["target"].split("/")[1]
        for ablation in d["ablations"]:
            if d["kind"] == "pretrain":
                print("\ngenerating tokens for", s, ablation)
                convert_dataset_hf_from_args(
                    dataset=d["target"],
                    data_subset=ablation,
                    splits=["train", "test"],
                    out_root=f"tokenized/{s}/{ablation}",
                    compression="zstd",
                    concat_tokens=2048,
                    tokenizer=tokenizer,
                    tokenizer_kwargs=None,
                    bos_text=None,
                    eos_text="<|endoftext|>",
                    no_wrap=False,
                    num_workers=None,
                )
            elif d["kind"] == "instruct":
                print(f"\nconvert_finetuning_dataset_from_args")
                convert_finetuning_dataset_from_args(
                    d["target"],
                    f"{ablation}",  # data_subset
                    ["train", "test"],
                    d["preproc"],
                    [],
                    False,
                    f"tokenized/{s}/{ablation}",  # out_root
                    None,
                    "zstd",
                    None,  # num_workers
                    "HuggingFaceTB/SmolLM2-135M",  # tokenizer
                    None,
                    20480,  # max_seq_len
                    "none",  # target_prompts
                    "last",  # target_responses
                    False,  # encoder_decoder
                )
            else:
                raise RuntimeError(f"Unknow dataset kind: {d['kind']}")


def create_upload(args, datasets):
    # upload all tokenized folders to corresponding repo/folder
    for s in args.source:
        d = datasets[s]
        print(f"Uploading {d['ablations']} from {d} to {d['target']} from {Path('.').absolute()}")
        for ablation in d["ablations"]:
            target_repo = d["target"]
            local_path = Path(".") / f"tokenized/{s}/{ablation}"
            print(f"\nUploading {ablation} to {target_repo} from {str(local_path)}\n")
            upload_token_folder(local_path, target_repo)
    print("upload finished.")

def upload_splits(args, datas):
    for arg in args.source:
        d = datas[arg]
        ds_name = d.get("ds_name", None)
        pull_n_push(
            d["target"],
            d["src"],
            ds_name=ds_name,
            ablations=d["ablations"],
            after_pull=d.get("after_pull", None),
        )


def main(args):
    datasets = {
        "tulu": {
            "src": "allenai/tulu-3-sft-olmo-2-mixture",
            "target": f"{args.target_repo}/split-tulu-3-sft-olmo-2-mixture",
            "after_pull": filter_tulu,
            "ablations": ("full", "100k", "10k", "1k"),
            "preproc":"preproc:pre_tulu",
            "kind": "instruct",
        },
        "numina": {
            "src": "AI-MO/NuminaMath-CoT",
            "target": f"{args.target_repo}/split-NuminaMath-CoT",
            "after_pull": process_numina,
            "ablations": ("full", "100k", "10k", "1k"),
            "preproc":"preproc:pre_numina",
            "kind": "instruct",
        },
        "finemath" :{
            "src": "HuggingFaceTB/finemath",
            "ds_name": "finemath-4plus",
            "target": f"{args.target_repo}/split-finemath",
            "ablations": ("full", "1M", "100k", "10k", "1k"),
            "kind": "pretrain",
        },
        "glaive": {
            "src": "glaiveai/glaive-code-assistant-v3",
            "target": f"{args.target_repo}/split-glaive-code-assistant-v3",
            "ablations": ("full", "100k", "10k", "1k"),
            "preproc":"preproc:pre_glaive",
            "kind": "instruct",
        },
        "avelinapythonedu": {
            "src": "Avelina/python-edu",
            "target": f"{args.target_repo}/split-avelina-python-edu",
            "ablations": ("full", "1M", "100k", "10k", "1k"),
            "kind": "pretrain",
        },
    }
    dataset_constants_split_config.register_new_datasets(args.target_repo)
    if args.split:
        print(f"spliting: {args.source}")
        d = upload_splits(args, datasets)
        print(f"spliting: {args.source} finished.")
    if args.tokenize:
        print(f"tokenizing: {args.source}")
        create_pretraining_tokens(args, datasets)
        print(f"tokenizing: {args.source} finished.")
    if args.upload:
        print(f"uploading tokens: {args.source}")
        create_upload(args, datasets)
        print(f"uploading tokens: {args.source} finished.")


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        "Split to train/test 1M, 100k, 10k, 1k and tokenize",
    )
    parser.add_argument(
        "--source",
        nargs="+",
        choices=["tulu", "numina", "finemath", "glaive", "avelinapythonedu",],
        default=["tulu", "numina", "finemath", "glaive", "avelinapythonedu"],
    )

    parser.add_argument(
        "--target_repo",
        default="LocalResearchGroup",
        help="target repo to upload splits and tokenizations",
    )

    parser.add_argument("--split", action=BooleanOptionalAction, default=True, help="split generation")
    parser.add_argument("--tokenize", action=BooleanOptionalAction, default=True, help="generate tokenization for splits")
    parser.add_argument("--upload", action=BooleanOptionalAction, default=True, help="upload tokenization folders")

    parsed = parser.parse_args()
    return parsed


if __name__ == "__main__":
    args = parse_args()
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("No Hugging Face token found. Please login.")
        login()
    main(args)
