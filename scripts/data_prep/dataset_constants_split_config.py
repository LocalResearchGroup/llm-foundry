from llmfoundry.command_utils import DatasetConstants, DataSplitConstants, add_dataset_config

def generate_constants(chars_per_sample, chars_per_token, label=None, splits=("full", 1, 10, 100, 1000)):
    ds_const = DatasetConstants(
        chars_per_sample=chars_per_sample,  # Computed over validation set
        chars_per_token=chars_per_token,  # OpenAI estimate
    )
    total_rows = None
    # we generate only train and test use --data_subset <xyzk> --out_root <defj>
    ds_const.splits[f"train"] = DataSplitConstants(
        hf_split="train",
        folder_split=f"train",
        raw_samples=total_rows,
        truncated_samples=total_rows,
    )

    ds_const.splits[f"test"] = DataSplitConstants(
        hf_split="test",
        folder_split=f"test",
        raw_samples=total_rows,
        truncated_samples=total_rows,
    )
    return ds_const


def register_new_datasets(target = "LocalResearchGroup"):
    _finemath = generate_constants(12163, 4)
    add_dataset_config(f"{target}/split-finemath", _finemath)
    _tulu = generate_constants(12163, 4)
    add_dataset_config(f"{target}/split-tulu-3-sft-olmo-2-mixture", _tulu)
    _numina = generate_constants(12163, 4)
    add_dataset_config(f"{target}/split-NuminaMath-CoT", _numina)
    _pythonedu = generate_constants(12163, 4)
    add_dataset_config(f"{target}/split-avelina-python-edu", _pythonedu)
    _glaive = generate_constants(12163, 4)
    add_dataset_config(f"{target}/split-glaive-code-assistant-v3", _glaive)
