"""Registration utilities for Llama models."""

from llmfoundry.models.llama.model import CustomLlamaModel


def get_custom_llama_model():
    """Get the CustomLlamaModel class."""
    return CustomLlamaModel


def register_custom_llama_model():
    """Register the custom Llama model with the registry."""
    from llmfoundry import registry
    registry.models.register("hf_causal_lm.custom_llama")(CustomLlamaModel)
    return CustomLlamaModel 