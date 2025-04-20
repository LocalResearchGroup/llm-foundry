# Local LLM Training Script

This is a local version of the LLM training script that runs directly on your GPUs without using Modal. It's designed to work with the LLM Foundry framework for training and fine-tuning language models.

## Prerequisites

1. **Python Environment**: Make sure you have Python 3.8+ installed with the necessary dependencies:
   - PyTorch
   - Transformers
   - Composer (MosaicML's training framework)
   - Flash Attention (optional, but recommended for performance)
   - HuggingFace Hub libraries

2. **GPU Requirements**: 
   - NVIDIA GPUs with CUDA support
   - Sufficient VRAM for the model you're training (at least 16GB recommended for Llama models)

3. **HuggingFace Access**:
   - A HuggingFace account with access to the models you want to use
   - HuggingFace token set in environment variables (HF_TOKEN, HUGGINGFACE_TOKEN, or HUGGINGFACE_HUB_TOKEN)

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

1. **Run the full training pipeline**:
   ```bash
   python local_llama_training.py
   ```

2. **Run individual functions**:
   You can also import and use individual functions from the script:
   ```python
   from local_llama_training import train_model, evaluate_model, generate_responses
   
   # Train a model
   run_ts = "20230101_120000"
   model_path = train_model(run_ts, yaml_path="scripts/train/yamls/llama/llama3-1b-lora2.yaml")
   
   # Evaluate the model
   evaluate_model(model_path)
   
   # Generate responses
   generate_responses(model_path, prompts=["Tell me a joke about programming"])
   ```

## Configuration

The script uses several constants at the top that you can modify:

- `PYTHON_PATH`: Path to your Python interpreter
- `TRAIN_DURATION`: How long to train for (in batches)
- `EVAL_INTERVAL`: How often to evaluate the model
- `SAVE_INTERVAL`: How often to save checkpoints
- `USE_CUSTOM_MODEL`: Whether to use the custom LlamaForCausalLM implementation
- `DATASET_BASE_PATH`: Where to store datasets
- `MODEL_CHECKPOINT_PATH`: Where to store model checkpoints
- `TRAIN_YAML`: Path to the training configuration YAML file
- `OUTPUT_PRECISION`: Precision for model outputs (bf16, fp16, etc.)

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

## Customizing Training

To customize the training process, modify the YAML configuration file specified by `TRAIN_YAML`. The default is `scripts/train/yamls/llama/llama3-1b-lora2.yaml`.

Key parameters you might want to adjust:
- Model size and architecture
- Learning rate and optimizer settings
- Batch size and training duration
- LoRA adapter configuration

## Troubleshooting

1. **Out of Memory Errors**:
   - Reduce batch size in the YAML configuration
   - Use gradient checkpointing
   - Try a smaller model

2. **Missing Dependencies**:
   - Make sure all required packages are installed
   - Check for version conflicts

3. **HuggingFace Access Issues**:
   - Verify your token is correctly set in environment variables
   - Check that you have access to the model you're trying to use

## License

This script is provided under the same license as the LLM Foundry project. 

Adapter pattern with inner and outer forward layers:
Yes, exactly right! This is a classic example of the Adapter pattern in software design:

### Inner Model (Core Computation)
- Created in _initialize_model_from_config as a basic nn.Module()

- Has its own  forward method defined and bound using forward.__get__(model)

- Contains the actual transformer architecture (embeddings, attention, MLP, etc.)
- Performs the raw computation of the language model
- Has weights copied from a HuggingFace model via _copy_weights_from_hf_llama

### Outer Model (Adapter/Wrapper)
- The CustomLlamaModel class that inherits from HuggingFaceModel


- Acts as an interface between your custom implementation and the training framework
- Its forward method:
  - Filters inputs to match what the inner model expects
  - Calls the inner model's forward
  - Standardizes outputs into a consistent format
- Adds framework-specific functionality like eval_forward and loss methods

This two-layer approach gives you the best of both worlds:

1. **Full control over the model architecture** - You can optimize or modify the inner implementation
2. **Framework compatibility** - The outer layer ensures it works with MosaicML's Composer
3. **PEFT support** - The adapter pattern makes it easy to apply PEFT techniques like LoRA
