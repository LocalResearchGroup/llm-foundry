"""Registration utilities for Llama models."""

from llmfoundry.models.llama.custom_model import CustomLlamaModel

def get_custom_llama_model():
    """Get the CustomLlamaModel class."""
    return CustomLlamaModel

def register_custom_llama_model():
    """Register the custom Llama model with the registry."""
    from llmfoundry import registry
    registry.models.register("smollm2-135m")(CustomLlamaModel)
    return CustomLlamaModel 