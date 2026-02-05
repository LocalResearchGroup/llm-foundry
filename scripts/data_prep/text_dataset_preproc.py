
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from datasets import load_dataset, load_from_disk, DatasetDict
from llmfoundry.data.finetuning.tasks import dataset_constructor
from data_lib.utils import get_datasets, register_new_datasets, _banner, str_rows_features


def create_refactor(dataset, decontaminated):
    ds_name = dataset["ds_name"] if "ds_name" in dataset else None
    process = dataset["after_pull"] if "after_pull" in dataset else None
    ds = dataset["decontaminated"] if decontaminated else dataset["original"]
    original = pull_orifinal_ds(ds, decontaminated, ds_name, process)
    return original
   
def pull_orifinal_ds(
    hf_ds_src,
    decontaminated,
    ds_name=None,
    after_pull=None,
):
    _banner(f"Loading dataset {hf_ds_src}/{'default' if ds_name is None else ds_name}")
    if ds_name: _banner(ds_name)
    from llmfoundry.command_utils.data_prep.convert_dataset_hf import CONSTS
    register_new_datasets()
    dataset = load_dataset(path=hf_ds_src, name=ds_name)
    if after_pull is not None:
        dataset = after_pull(dataset, decontaminated)
    return dataset


#% main loop
def _main_loop(args):
    ds_config = get_datasets()
    # Add after pull call to process instruct datasets with template
    ds_config["tulu"]["after_pull"] = filter_tulu
    ds_config["numina"]["after_pull"] = process_numina
    ds_config["glaive"]["after_pull"] = process_glaive
    for ds in args.datasets:
        dataset = create_refactor(ds_config[ds], args.decontaminated)
        private=False
        hf_repo = f"{args.user_org}/{ds}-with-template{'-decontaminated' if args.decontaminated else ''}"
        label="default"
        shard_size = "128MB"
        dataset.push_to_hub(hf_repo, config_name=label, private=private, max_shard_size=shard_size)


#% chat ml template and filtering of original datasets
def apply_chatml_template(inp: dict, k_prompt: str, k_response: str):
    """Format dataset into ChatML template."""
    prompt = (
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Local Research Group<|im_end|>\n"
        f"<|im_start|>user\n{inp[k_prompt]}\n<|im_end|>\n"
    )
    response = (
        f"<|im_start|>assistant\n{inp[k_response]}<|im_end|>\n"
        "<|endoftext|>"
    )
    return {"prompt": prompt, "response": response}


def template_to_tulu(inp: dict):
    return apply_chatml_template(inp, "prompt", "response")


def template_to_numina(inp: dict):
    return apply_chatml_template(inp, "problem", "solution")


def template_to_glaive(inp: dict):
    return apply_chatml_template(inp, "question", "answer")


def filter_tulu(dataset, decontaminated):
    print(f"\n\ntulu {str_rows_features(dataset)}\n\n")
    if not decontaminated:
        dataset = dataset.filter(lambda r: r["source"] is not None and "aya" not in r["source"] and len(r["messages"]) == 2)
        dataset = dataset.remove_columns(["source", "dataset"])
    dataset = dataset.remove_columns(["id"])

    def extract_qa(messages):
        user_question = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
        assistant_response = next((msg["content"] for msg in messages if msg["role"] == "assistant"), None)
        return {"prompt": user_question, "response": assistant_response}

    # Apply function to dataset
    dataset = dataset.map(lambda example: extract_qa(example["messages"])) if not decontaminated else dataset
    dataset = dataset.remove_columns(["messages"]) if not decontaminated else dataset
    dataset = dataset.map(lambda example: template_to_tulu(example)) if not decontaminated else dataset
    print(f"tulu after {str_rows_features(dataset)}")
    return dataset


def process_numina(dataset, decontaminated):
    print(f"numina {str_rows_features(dataset)}")
    # remove conflictlict that breaks pytorch collate with 2 row per batch!
    dataset = dataset.map(lambda example: template_to_numina(example))
    colums = ["source", "problem", "solution"]
    if not decontaminated: colums.append("messages")
    dataset = dataset.remove_columns(colums)
    print(f"numina processed: {str_rows_features(dataset)}")
    return dataset


def process_glaive(dataset, decontaminated):
    print(f"glaive {str_rows_features(dataset)}")

    def extract_qa(messages):
        return template_to_glaive(messages)

    dataset = dataset.map(lambda example: extract_qa(example))
    dataset = dataset.remove_columns(["question", "answer"])
    print(f"glaive processed: {str_rows_features(dataset)}")

    return dataset

#% argument parsing section
def main(args):
    if args.datasets:
        _main_loop(args)


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description="""Refactor instruct datasets with and witout decontamination
        """,
    )
    ds = [k for k in get_datasets() if get_datasets()[k]["kind"] == "instruct"]

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=ds,
        default=ds,
    )

    parser.add_argument(
        "--user_org",
        default="LocalResearchGroup",
        help="user/org base namespace default is `LocalResearchGroup`",
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
    import os
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("No Hugging Face token found. Please login.")
        login()
    main(args)

