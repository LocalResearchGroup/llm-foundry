from llmfoundry.data.finetuning.tasks import (
    DatasetConstructor,
)

HF_REPO="LocalResearchGroup"
dataset_constructor = DatasetConstructor()

# @dataset_constructor.register(f"{HF_REPO}/split-tulu-3-sft-olmo-2-mixture")
# def pre_tulu(inp: dict):
#     return {'prompt': inp["prompt"], 'response': inp["response"]}


# @dataset_constructor.register(f"{HF_REPO}/split-NuminaMath-CoT")
# def pre_numina(inp: dict):
#     return {'prompt': inp['problem'], 'response': inp['solution']}


# @dataset_constructor.register(f"{HF_REPO}/split-glaive-code-assistant-v3")
# def pre_glaive(inp: dict):
#     return {'prompt': inp['question'], 'response': inp['answer']}



def preproc_chatml(inp: dict, k_prompt:str, k_response: str):
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

@dataset_constructor.register(f"{HF_REPO}/split-tulu-3-sft-olmo-2-mixture")
def pre_ml_tulu(inp: dict):
    return preproc_chatml(inp, "prompt", "response")


@dataset_constructor.register(f"{HF_REPO}/split-NuminaMath-CoT")
def pre_ml_numina(inp: dict):
    return preproc_chatml(inp, "problem", "solution")


@dataset_constructor.register(f"{HF_REPO}/split-glaive-code-assistant-v3")
def pre_ml_glaive(inp: dict):
    return preproc_chatml(inp, "question", "answer")


