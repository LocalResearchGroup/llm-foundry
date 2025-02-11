# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Callable

import pytest
from pytest import fixture
from transformers import PreTrainedTokenizerBase

from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from llmfoundry.utils.builders import build_composer_model, build_tokenizer


def _build_model(config: dict[str, Any], tokenizer: PreTrainedTokenizerBase):
    name = config.pop("name")
    model = build_composer_model(
        name=name,
        cfg=config,
        tokenizer=tokenizer,
    )
    return model


def tiny_gpt2_model_helper(config):  # type: ignore
    transformers = pytest.importorskip("transformers")

    return transformers.AutoModelForCausalLM.from_config(config)


@pytest.fixture(scope="session")
def _session_tiny_gpt2_model(_session_tiny_gpt2_config):  # type: ignore
    return tiny_gpt2_model_helper(_session_tiny_gpt2_config)


def tiny_gpt2_config_helper():
    transformers = pytest.importorskip("transformers")

    tiny_overrides = {
        "n_embd": 2,
        "n_head": 2,
        "n_layer": 2,
        "vocab_size": 50258,  # 50257 + 1 for pad token
    }
    return transformers.AutoConfig.from_pretrained("gpt2", **tiny_overrides)


@pytest.fixture(scope="session")
def _session_tiny_gpt2_config():  # type: ignore
    return tiny_gpt2_config_helper()


def tiny_gpt2_tokenizer_helper():
    transformers = pytest.importorskip("transformers")

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return hf_tokenizer


@pytest.fixture
def tiny_gpt2_model(_session_tiny_gpt2_model):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_model)


@pytest.fixture(scope="session")
def _session_tiny_gpt2_tokenizer():  # type: ignore
    return tiny_gpt2_tokenizer_helper()


@pytest.fixture
def tiny_gpt2_tokenizer(_session_tiny_gpt2_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_gpt2_tokenizer)


def tiny_llama_tokenizer_helper():
    transformers = pytest.importorskip("transformers")

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "huggyllama/llama-7b",
        use_fast=False,
    )
    return hf_tokenizer


@pytest.fixture(scope="session")
def _session_tiny_llama_tokenizer():  # type: ignore
    return tiny_llama_tokenizer_helper()


@pytest.fixture
def tiny_llama_tokenizer(_session_tiny_llama_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_llama_tokenizer)


def tiny_opt_tokenizer_helper():
    transformers = pytest.importorskip("transformers")

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "facebook/opt-125m",
    )
    hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return hf_tokenizer


def tiny_opt_model_helper(config):  # type: ignore
    transformers = pytest.importorskip("transformers")

    return transformers.AutoModelForCausalLM.from_config(config)


@pytest.fixture(scope="session")
def _session_tiny_opt_tokenizer():  # type: ignore
    return tiny_opt_tokenizer_helper()


@pytest.fixture(scope="session")
def _session_tiny_opt_config():  # type: ignore
    return tiny_opt_config_helper()


@pytest.fixture(scope="session")
def _session_tiny_opt_model(_session_tiny_opt_config):  # type: ignore
    return tiny_opt_model_helper(_session_tiny_opt_config)


def tiny_opt_config_helper():
    transformers = pytest.importorskip("transformers")

    tiny_overrides = {
        "n_embd": 2,
        "n_head": 2,
        "n_layer": 2,
        "vocab_size": 50272,
    }
    return transformers.AutoConfig.from_pretrained(
        "facebook/opt-125m",
        **tiny_overrides,
    )


@pytest.fixture
def tiny_opt_tokenizer(_session_tiny_opt_tokenizer):  # type: ignore
    return copy.deepcopy(_session_tiny_opt_tokenizer)


@pytest.fixture
def tiny_opt_model(_session_tiny_opt_model):  # type: ignore
    return copy.deepcopy(_session_tiny_opt_model)
