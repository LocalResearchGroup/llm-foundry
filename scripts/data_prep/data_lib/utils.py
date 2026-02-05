#% utils
def _banner(msg):
    print("#"*len(msg))
    print(msg)
    print("#"*len(msg))

def str_rows_features(ds):
    return f"rows: {len(ds)} features: {ds['train'].features}"

def get_datasets():  # target_repo):
    ds_config = {
        "tulu": {
            "original": "allenai/tulu-3-sft-olmo-2-mixture",
            "decontaminated": "LocalResearchGroup/split-tulu-3-sft-olmo-2-mixture-decontaminated",
            "kind": "instruct",
            "template": "tulu-with-template",
        },
        "numina": {
            "original": "AI-MO/NuminaMath-CoT",
            "decontaminated": "LocalResearchGroup/split-NuminaMath-CoT-decontaminated",
            "kind": "instruct",
            "template": "numina-with-template",
        },
        "glaive": {
            "original": "glaiveai/glaive-code-assistant-v3",
            "decontaminated": "LocalResearchGroup/split-glaive-code-assistant-v3-decontaminated",
            "kind": "instruct",
            "template": "glaive-with-template",
        },
        "finemath": {
            "original": "HuggingFaceTB/finemath",
            "decontaminated": "LocalResearchGroup/split-finemath-decontaminated",
            "kind": "pretrain",
            "ds_name": "finemath-4plus",
        },
        "pythonedu": {
            "original": "Avelina/python-edu",
            "decontaminated": "LocalResearchGroup/split-avelina-python-edu-decontaminated",
            "kind": "pretrain",
        },
    }
    return ds_config


def rel_path(name, decontaminated):
    return f"{name}" \
    f"{'-with-template' if get_datasets()[name]['kind'] == 'instruct' else ''}" \
    f"{'-decontaminated' if decontaminated else ''}"

#% Allow to add extra datasets to CONSTS

def add_dataset_config(name, splits):
    from llmfoundry.command_utils.data_prep.convert_dataset_hf import CONSTS
    CONSTS[name] = splits


def generate_constants(total_rows, chars_per_sample, chars_per_token):
    from llmfoundry.command_utils.data_prep.convert_dataset_hf import CONSTS, DataSplitConstants, DatasetConstants

    ds_const = DatasetConstants(
        chars_per_sample=chars_per_sample,
        chars_per_token=chars_per_token,
    )
    ds_const.splits["train"] = DataSplitConstants(
        hf_split="train",
        folder_split="train",
        raw_samples=total_rows,
        truncated_samples=None,
    )

    ds_const.splits["test"] = DataSplitConstants(
        hf_split="test",
        folder_split="test",
        raw_samples=total_rows,
        truncated_samples=None,
    )
    return ds_const


def register_new_datasets(target = "LocalResearchGroup"):
    constants = {
        "finemath": generate_constants(6_700_000, 6212, 4),
        "tulu": generate_constants(939_000, 6212, 4),
        "numina": generate_constants(859_00, 6212, 4),
        "pythonedu": generate_constants(7_680_000, 6212, 4),
        "glaive": generate_constants(950_000, 6212, 4),
    }
    ds = get_datasets()
    for name in ds.keys():
        add_dataset_config(ds[name]["original"], constants[name])
        add_dataset_config(ds[name]["decontaminated"], constants[name])

