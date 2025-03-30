from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from huggingface_hub import HfApi, login
import os


def main(args):
    api = HfApi()
    datasets = {
        "tulu": {
            "target": f"{args.repo}/split-tulu-3-sft-olmo-2-mixture",
            "ablations": ("full", "100k", "10k", "1k"),
        },
        "numina": {
            "target": f"{args.repo}/split-NuminaMath-CoT",
            "ablations": ("full", "100k", "10k", "1k"),
        },
        "finemath" :{
            "target": f"{args.repo}/split-finemath",
            "ablations": ("full", "1M", "100k", "10k", "1k"),
        }
    }
    datas_list = args.dataset
    
    from pprint import pp
    pp(datasets)
    for ds in datas_list:
        print(f"downloading {datasets[ds]["target"]=} to download-{ds}-tokenized\n")
        local_dir = api.snapshot_download(
            repo_id=datasets[ds]["target"],
            repo_type="dataset",
            local_dir=f"download-{ds}-tokenized",
        )

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Downloads tokenized versions of train/test 1M, 100k, 10k, 1k',
    )
    parser.add_argument(
        '--dataset',
        nargs='+',
        choices=['tulu', 'numina', 'finemath'],
        default=['tulu', 'numina', 'finemath'],
    )

    parser.add_argument(
        "--repo",
        default="LocalResearchGroup",
        help="repo containing tokenizations",
    )
        
    parsed = parser.parse_args()
    return parsed


if __name__ == "__main__":
    args = parse_args()
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("No Hugging Face token found. Please login.")
        login()
    main(args)