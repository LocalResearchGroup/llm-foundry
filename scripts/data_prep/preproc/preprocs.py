from llmfoundry.data.finetuning.tasks import (
    DatasetConstructor,
)

dataset_constructor = DatasetConstructor()

@dataset_constructor.register(f"LocalResearchGroup/split-tulu-3-sft-olmo-2-mixture")
def pre_ml_tulu(inp: dict):
    return {"prompt": inp["prompt"], "response": inp["response"]}


@dataset_constructor.register(f"LocalResearchGroup/split-NuminaMath-CoT")
def pre_ml_numina(inp: dict):
    return {"prompt": inp["prompt"], "response": inp["response"]}


@dataset_constructor.register(f"LocalResearchGroup/split-glaive-code-assistant-v3")
def pre_ml_glaive(inp: dict):
    return {"prompt": inp["prompt"], "response": inp["response"]}


