from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from huggingface_hub import HfApi, login
import os


def main(args):
    api = HfApi()
    datasets = {
        "tulu": {
            "target": f"{args.repo}/split-tulu-3-sft-olmo-2-mixture",
        },
        "numina": {
            "target": f"{args.repo}/split-NuminaMath-CoT",
        },
        "finemath" :{
            "target": f"{args.repo}/split-finemath",
        },
        "glaive" : {
            "target": f"{args.repo}/split-glaive-code-assistant-v3",
        },
        "avelinapythonedu": {
            "target": f"{args.repo}/split-avelina-python-edu",
        },
    }
    
    for ds in args.dataset:
        ld = f"{args.out}/{ds}"
        datadown = datasets[ds]["target"]
        print(f"downloading {datadown=} to {ld=}\n")
        local_dir = api.snapshot_download(
            repo_id=datadown,
            repo_type="dataset",
            local_dir=ld,
        )

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        "Downloads tokenized versions of train/test 1M, 100k, 10k, 1k",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["tulu", "numina", "finemath", "glaive", "avelinapythonedu"],
        default=["tulu", "numina", "finemath", "glaive", "avelinapythonedu"],
    )

    parser.add_argument(
        "--repo",
        default="LocalResearchGroup",
        help="repo containing tokenizations",
    )

    parser.add_argument(
        "--out",
        default=".",
        help="local download folder",
    )
        
    parsed = parser.parse_args()
    return parsed


if __name__ == "__main__":
    args = parse_args()
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("No Hugging Face token found. Please login.")
        login()
    main(args)