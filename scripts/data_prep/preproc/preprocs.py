from llmfoundry.data.finetuning.tasks import (
    DatasetConstructor,
)

dataset_constructor = DatasetConstructor()

@dataset_constructor.register("tyoc213/split-finemath")
def preprocessing_function(inp: dict) -> dict:
    """Format the already-split example."""
    return {
        'prompt': inp['inputs'] + ':',
        'response': inp['targets'],
    }

@dataset_constructor.register("tyoc213/split-tulu-3-sft-olmo-2-mixture")
def pre_tulu(inp: dict):
    pro = [m["content"] for m in inp["messages"] if m["role"] == "user"]
    ans = [m["content"] for m in inp["messages"] if m["role"] == "assistant"]

    prompt = "\n".join(pro)
    answer = "\n".join(ans)
    if len(prompt) < len(answer):
        # FIXME: padding needed if not we get an error?
        prompt = prompt + "  " + " " * (len(answer) - len(prompt)) 
    return {'prompt': prompt, 'response': answer}


@dataset_constructor.register("tyoc213/split-NuminaMath-CoT")
def pre_numina(inp: dict):
    return {'prompt': inp['problem'], 'response': inp['solution']}
