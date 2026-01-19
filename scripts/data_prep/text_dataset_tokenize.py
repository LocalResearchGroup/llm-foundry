from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from huggingface_hub import HfApi, login
from pathlib import Path

import os


from convert_finetuning_dataset import convert_finetuning_dataset_from_args
from llmfoundry.command_utils import convert_dataset_hf_from_args

from data_lib.utils import get_datasets, _banner, register_new_datasets, rel_path

#%

def upload_token_folder(folder_path, namespace, path_in_repo):
    _banner(f"Uploading {folder_path} to {namespace}")
    api = HfApi()
    cr = api.create_repo(namespace, repo_type="dataset", exist_ok=True)
    r = api.upload_folder(
        repo_id=namespace,
        repo_type="dataset",
        folder_path=folder_path,
        path_in_repo=path_in_repo,
    )
    print(f"token uploaded result: {r}@{cr}")



def create_tokenized_upload(name, user_org, decontaminated):
    dataset = get_datasets()[name]
    ablations = ["train", "test"] if decontaminated else ["train"]
    data_subset = dataset["ds_name"] if "ds_name" in dataset else "default"
    if name in ["finemath"] and decontaminated:
        data_subset = "default"
    for ablation in ablations:
        namespace = f"{user_org}/{rel_path(name, decontaminated)}-tokenized"
        local_path = Path(".") / f"tokenized/{rel_path(name, decontaminated)}/{data_subset}/{ablation}"  # out_root
        upload_token_folder(local_path, namespace, f"/{ablation}")
    print("upload finished.")

def create_tokens(name, user_org, decontaminated):
    dataset = get_datasets()[name]
    max_seq_len = 8192
    data_subset = dataset["ds_name"] if "ds_name" in dataset else "default"
    if name in ["finemath"] and decontaminated:
        data_subset = "default"

    if dataset["kind"] == "pretrain":
        print("\nconvert_dataset_hf_from_args for", name, data_subset)
        print(f"{dataset['decontaminated'] if decontaminated else dataset['original']}\n\n")
        tokenizer="HuggingFaceTB/SmolLM2-135M"
        convert_dataset_hf_from_args(
            dataset=f"{dataset['decontaminated'] if decontaminated else dataset['original']}",
            data_subset=data_subset,
            splits=["train", "test"] if decontaminated else ['train'],
            out_root=f"tokenized/{rel_path(name, decontaminated)}/{data_subset}",
            compression="zstd",
            concat_tokens=max_seq_len,
            tokenizer=tokenizer,
            tokenizer_kwargs=f'{{"model_max_length": {max_seq_len} }}',
            bos_text=None,
            eos_text="<|endoftext|>",
            no_wrap=True,
            num_workers=None,
        )
    elif dataset["kind"] == "instruct":
        print(f"\nconvert_finetuning_dataset_from_args for", data_subset)
        print(f"{user_org}/{rel_path(name,decontaminated)}\n\n")
        tokenizer="HuggingFaceTB/SmolLM2-135M-instruct"
        convert_finetuning_dataset_from_args(
            f"{user_org}/{rel_path(name,decontaminated)}",
            ###### f"{dataset['decontaminated'] if decontaminated else dataset['original']}",
            f"{data_subset}",  # data_subset
            ["train", "test"] if decontaminated else ['train'],
            None,  # no preprocessing dataset is ready
            [],
            True,
            f"tokenized/{rel_path(name, decontaminated)}/{data_subset}",  # out_root
            None,
            "zstd",
            None,  # num_workers
            tokenizer,  # tokenizer
            None,
            max_seq_len,  # max_seq_len
            "none",  # target_prompts
            "last",  # target_responses
            False,  # encoder_decoder
        )
    else:
        raise RuntimeError(f"Unknow dataset kind: {d['kind']}")



#% main loop
def main(args):
    register_new_datasets()
    for ds in args.datasets:
        if args.tokenize:
            _banner(f"Making tokens for {ds} {args.decontaminated}")
            dataset = create_tokens(ds, args.user_org, args.decontaminated)
        if args.upload_tokens:
            _banner(f"Uploading tokens for {ds} {args.decontaminated}")
            create_tokenized_upload(ds, args.user_org, args.decontaminated)


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description="""Tool to refactor instruct datasets
        """,
    )
    datasets = get_datasets()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=datasets.keys(),
        default=datasets.keys()
    )

    parser.add_argument(
        "--user_org",
        default="LocalResearchGroup",
        help="user/org base namespace to upload tokens default is `LocalResearchGroup`",
    )

    parser.add_argument(
        "--tokenize",
        action=BooleanOptionalAction,
        default=True,
        help="generate local tokenization for splits",
    )
    parser.add_argument(
        "--upload-tokens",
        action=BooleanOptionalAction,
        default=True,
        help="upload local tokenization to user/org",
    )

    parser.add_argument(
        "--decontaminated",
        action=BooleanOptionalAction,
        default=False,
        help="use decontaminated dataset instead of original one",
    )


    parsed = parser.parse_args()
    return parsed


if __name__ == "__main__":
    args = parse_args()
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("No Hugging Face token found. Please login.")
        login()
    main(args)

