from llmfoundry.command_utils import convert_dataset_hf_from_args, DatasetConstants, DataSplitConstants, add_dataset_config, CONSTS
from llmfoundry.command_utils import convert_dataset_hf_from_args, DatasetConstants, DataSplitConstants, add_dataset_config, CONSTS

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

_finemath = generate_constants(2163, 4)
HF_TARGET = "tyoc213"  # "LocalResearchGroup"
add_dataset_config(f"{HF_TARGET}/split-finemath", _finemath)
_tulu = generate_constants(2163, 4)
add_dataset_config(f"{HF_TARGET}/split-tulu-3-sft-olmo-2-mixture", _tulu)
_numina = generate_constants(2163, 4)
add_dataset_config(f"{HF_TARGET}/split-NuminaMath-CoT", _numina)
_pythonedu = generate_constants(2163, 4)
add_dataset_config(f"{HF_TARGET}/split-avelina-python-edu", _pythonedu)


# def do_xxx():
#     args = Namespace(dataset='tyoc213/split-finemath', data_subset='10k', splits=['train', 'test'], out_root='finemath-10k', compression=None, concat_tokens=None, tokenizer='HuggingFaceTB/SmolLM2-135M', tokenizer_kwargs=None, bos_text=None, eos_text='<|endoftext|>', no_wrap=False, num_workers=None)
#     convert_dataset_hf_from_args(
#         dataset=args.dataset,
#         data_subset=args.data_subset,
#         splits=args.splits,
#         out_root=args.out_root,
#         compression=args.compression,
#         concat_tokens=args.concat_tokens,
#         tokenizer=args.tokenizer,
#         tokenizer_kwargs=args.tokenizer_kwargs,
#         bos_text=args.bos_text,
#         eos_text=args.eos_text,
#         no_wrap=args.no_wrap,
#         num_workers=args.num_workers,
#     )


# if __name__ == "__main__":
#     do_xxx()