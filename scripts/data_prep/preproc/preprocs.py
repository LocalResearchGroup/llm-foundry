from llmfoundry.data.finetuning.tasks import (
    DatasetConstructor,
)

dataset_constructor = DatasetConstructor()


def preproc_chatml(inp: dict, k_prompt:str, k_response: str):
    """Format dataset into ChatML template."""
    return {"prompt": inp[k_prompt], "response": inp[k_response]}

@dataset_constructor.register(f"LocalResearchGroup/split-tulu-3-sft-olmo-2-mixture")
def pre_ml_tulu(inp: dict):
    return preproc_chatml(inp, "prompt", "response")


@dataset_constructor.register(f"LocalResearchGroup/split-NuminaMath-CoT")
def pre_ml_numina(inp: dict):
    return preproc_chatml(inp, "problem", "solution")


@dataset_constructor.register(f"LocalResearchGroup/split-glaive-code-assistant-v3")
def pre_ml_glaive(inp: dict):
    return preproc_chatml(inp, "question", "answer")


