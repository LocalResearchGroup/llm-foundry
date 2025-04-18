#!/usr/bin/env python
"""Train a model using the CustomLlamaModel."""

from llmfoundry.train import train
from llmfoundry.utils.config_utils import load_config


def main():
    """Main function."""
    # Load the configuration
    config_path = "configs/train_custom_llama.yaml"
    config = load_config(config_path)
    
    # Get the CustomLlamaModel class and register it
    from llmfoundry.models.llama.register import register_custom_llama_model
    register_custom_llama_model()
    
    # Train the model
    train(config)


if __name__ == "__main__":
    main() 