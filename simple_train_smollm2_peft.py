#!/usr/bin/env python3
"""Simplified PEFT training script for custom SmolLM2-135M model."""

import os
import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from llmfoundry.models.llama.register import register_custom_llama_model
from llmfoundry.command_utils.train import train
from omegaconf import OmegaConf

import text_generation_callback  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main PEFT training function."""
    logger.info("Registering custom SmolLM2-135M model...")
    register_custom_llama_model()
    logger.info("Custom model registered successfully!")
    
    config_path = "scripts/train/yamls/pretrain/custom_smollm2-135m_peft.yaml"
    logger.info(f"Loading configuration from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    save_folder = "model-checkpoints/custom_smollm2_135m_peft"
    config.save_folder = save_folder
    os.makedirs(save_folder, exist_ok=True)
    logger.info(f"PEFT model checkpoints will be saved to: {save_folder}")
    
    dataset_local = config.variables.data_local
    dataset_remote = getattr(config.variables, 'data_remote', None)
    if dataset_remote and str(dataset_remote).strip():
        os.makedirs(dataset_local, exist_ok=True)
        logger.info(
            f"Streaming dataset from remote: {dataset_remote} with local cache: {dataset_local}")
    else:
        if not os.path.exists(dataset_local):
            logger.warning(f"Dataset not found at: {dataset_local}")
            return
        logger.info(f"Using local dataset at: {dataset_local}")
    
    if hasattr(config.model, 'peft_config') and config.model.peft_config:
        peft_config = config.model.peft_config
        logger.info("PEFT Configuration:")
        logger.info(f"  - Type: {peft_config.peft_type}")
        logger.info(f"  - Rank (r): {peft_config.r}")
        logger.info(f"  - Alpha: {peft_config.lora_alpha}")
        logger.info(f"  - Dropout: {peft_config.lora_dropout}")
        logger.info(f"  - Target modules: {peft_config.target_modules}")
        logger.info(f"  - Use RSLora: {peft_config.get('use_rslora', False)}")
        logger.info(f"  - Use DoRA: {peft_config.get('use_dora', False)}")
    else:
        logger.warning("No PEFT configuration found!")
        return
    
    logger.info("Starting PEFT training...")
    try:
        trainer = train(config)
        logger.info("PEFT training completed successfully!")
        logger.info(f"PEFT adapters saved to: {save_folder}")
        return trainer
    except Exception as e:
        logger.error(f"PEFT training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()