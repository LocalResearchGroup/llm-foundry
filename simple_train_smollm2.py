#!/usr/bin/env python3
"""
Simplified training script for custom SmolLM2-135M model.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llmfoundry.models.llama.register import register_custom_llama_model
from llmfoundry.command_utils.train import train
from omegaconf import OmegaConf

# Import the text generation callback to register it
import text_generation_callback  # type: ignore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    
    # Register the custom model
    logger.info("Registering custom SmolLM2-135M model...")
    register_custom_llama_model()
    logger.info("Custom model registered successfully!")
    
    # Load the training configuration
    config_path = "scripts/train/yamls/pretrain/custom_smollm2-135m.yaml"
    logger.info(f"Loading configuration from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Set save folder
    save_folder = "model-checkpoints/custom_smollm2_135m"
    config.save_folder = save_folder
    os.makedirs(save_folder, exist_ok=True)
    logger.info(f"Model checkpoints will be saved to: {save_folder}")
    
    # Verify dataset path
    # Install with:
    # python scripts/data_prep/convert_dataset_hf.py --dataset allenai/c4 --data_subset en --out_root datasets/c4_small \
    # --splits train_small val_small --concat_tokens 2048 --tokenizer HuggingFaceTB/SmolLM2-135M \
    # --eos_text '<|endoftext|>' --compression zstd
    dataset_path = config.variables.data_local
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset not found at: {dataset_path}")
        logger.info("Please ensure the C4 dataset is available in the datasets/c4_small directory")
        logger.info("You can download it or create a symbolic link to the correct location")
        return
    
    logger.info(f"Using dataset at: {dataset_path}")
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer = train(config)
        logger.info("Training completed successfully!")
        return trainer
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 