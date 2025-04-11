# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""
from argparse import ArgumentParser, Namespace

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


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert dataset into MDS format, optionally concatenating and tokenizing',
    )
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument(
        '--data_subset',
        type=str,
        default=None,
        help='E.g. "all" or "en"',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'train_small', 'val', 'val_small', 'val_xsmall'],
    )
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--compression', type=str, default=None)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--concat_tokens',
        type=int,
        help='Convert text to tokens and concatenate up to this many tokens',
    )

    parser.add_argument('--tokenizer', type=str, required=False, default=None)
    parser.add_argument('--tokenizer_kwargs', type=str, required=False)
    parser.add_argument('--bos_text', type=str, required=False, default=None)
    parser.add_argument('--eos_text', type=str, required=False, default=None)
    parser.add_argument('--no_wrap', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, required=False, default=None)

    parsed = parser.parse_args()
    return parsed


if __name__ == '__main__':
    args = parse_args()
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
