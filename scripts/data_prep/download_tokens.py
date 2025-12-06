from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from huggingface_hub import HfApi, login
import os

from data_lib.utils import get_datasets, rel_path


TRAINING_GPU = os.environ.get("MODAL_GPU", "L4") 

def main(args):
    api = HfApi()
    
    for ds in args.datasets:
        ld = f"{args.out}/{ds}"
        datadown = f"{args.user_org}/{rel_path(ds, args.decontaminated)}-tokenized"
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
    datasets = get_datasets().keys()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=datasets,
        default=datasets,
    )

    parser.add_argument(
        "--user_org",
        default="LocalResearchGroup",
        help="user/org containing tokenizations",
    )

    parser.add_argument(
        "--out",
        default=".",
        help="local download folder",
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
