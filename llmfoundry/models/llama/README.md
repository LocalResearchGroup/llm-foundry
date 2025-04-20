# Training Custom Llama Models

## Customizing Training

### YAML file
To customize the training process, modify the YAML configuration file specified by `TRAIN_YAML`. The default is `scripts/train/yamls/llama/llama3-1b-lora2.yaml`.

### train_with_custom_llama.py

train_with_custom_llama.py serves as the entry point for training with our custom LLaMA implementation. It handles the configuration loading from YAML files, registers our CustomLlamaModel with the model registry, and orchestrates the training process. The script manages critical setup tasks including HuggingFace authentication, dataset path configuration, and preparing model parameters before delegating to the training framework. It can be customized through command-line arguments or environment variables, making it flexible for different training scenarios.

### Weight Loading in CustomLlamaModel

The  _copy_weights_from_hf_llama method handles weight transfer from standard Hugging Face models to our custom implementation. It first loads a Hugging Face model via from_pretrained() to serve as a source, then systematically copies weights component by component including embeddings, transformer layers, normalization layers and output head. The method explicitly tracks copy progress, reporting both successful transfers and any uninitialized weights to ensure model integrity. This direct weight mapping approach enables our custom implementation to precisely match pretrained model behavior while gaining the performance benefits of our optimized architecture.


### CustomLlamaModel Initialization and Adapter Pattern

CustomLlamaModel follows a two-layer architecture that separates model implementation from framework integration. The outer class inherits from HuggingFaceModel, managing compatibility with the training framework, while the inner model (created via _initialize_model_from_config) implements the actual transformer architecture with optimized components. During initialization, the class loads a pretrained model, creates a corresponding optimized implementation, then systematically transfers weights via _copy_weights_from_hf_llama. This adapter pattern allows for performance optimizations in the inner model while maintaining full compatibility with HuggingFace's ecosystem, and includes built-in support for PEFT adapters that can be attached to the initialized model.


### Dual Forward Methods in the Adapter Pattern

The CustomLlamaModel implements two distinct forward methods that operate in tandem. The inner model's forward method (bound to the model instance using forward.__get__) contains the raw computational logic for the transformer architecture, handling token embeddings, attention operations, and feed-forward networks. The outer CustomLlamaModel's forward method serves as an adapter interface, filtering input arguments to match inner model requirements, managing state tracking, and implementing training-specific logic like loss calculation via the fused loss function. This separation allows the inner model to remain focused on efficient computation while the outer wrapper handles framework integration, creating a clean division of responsibilities that simplifies maintenance and optimization.

### Model Registration and Framework Integration

The register_custom_llama_model() function in register.py integrates our custom model implementation with the training framework. It adds the CustomLlamaModel class to the framework's model registry under the key "hf_causal_lm", allowing our model to be used wherever HuggingFace causal language models are supported. This registration happens explicitly in both train_with_custom_llama.py before starting training and in local_llama_training.py's evaluate_model function before evaluation begins. Without this registration step, the framework would use a standard implementation instead of our optimized version with custom components.

### local_llama_training.py

The local script adapts the Modal cloud deployment approach for single-machine environments while preserving the core workflow. Key differences include file path handling (local directories vs Modal Volumes), environment setup (local Python interpreter vs containerized environment), and execution model (synchronous function calls vs Modal's distributed functions). The local script adds more comprehensive logging, path validation, and error handling to manage filesystem interactions that Modal handles automatically. While Modal's script leverages cloud-specific features like network tunneling for Aim visualization and GPU provisioning via decorators, the local version provides equivalent functionality through direct subprocess calls and environment variable configuration. The way custom model integration happens should not change. 

This is a local version of the LLM training script that runs directly on your GPUs without using Modal. It's designed to work with the LLM Foundry framework for training and fine-tuning language models.

## Prerequisites

**Follow the steps to install llmfoundry**

## Setup

1. **Clone the LLM Foundry repository**:
   ```bash
   git clone https://github.com/mosaicml/llm-foundry.git
   cd llm-foundry
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Set up your HuggingFace token**:
   ```bash
   export HF_TOKEN=your_token_here
   ```

## Usage

**Run the full training pipeline**:
   ```bash
   python local_llama_training.py
   ```

## Directory Structure

The script creates the following directory structure:

```
./
├── datasets/              # Dataset storage
│   └── c4_small/          # C4 dataset
├── model-checkpoints/     # Model checkpoints
├── runs/                  # Training run outputs
│   └── model-name-timestamp/  # Individual run
└── local_llama_training.py  # This script
```

