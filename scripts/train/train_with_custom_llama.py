import os
from pathlib import Path
import logging
from typing import Optional
from omegaconf import OmegaConf

from llmfoundry.models.llama import CustomLlamaModel
from llmfoundry.registry import models
from llmfoundry.command_utils.train import train
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the script's directory
ROOT_DIR = Path(__file__).parent.parent.parent


def train_custom_llama(
    model_name: Optional[str] = None,
    yaml_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    dataset_path: Optional[str] = None,
):
    """Train a custom Llama model using the specified configuration."""
    try:
        # Set up paths
        if yaml_path is None:
            yaml_path = os.path.join(ROOT_DIR, "scripts", "train", "yamls", "llama", "llama3-1b-lora.yaml")
        if output_dir is None:
            output_dir = os.path.join(ROOT_DIR, "outputs/custom_llama")
        if dataset_path is None:
            dataset_path = os.path.join(ROOT_DIR, "datasets/c4_small")
            logger.info(f"Using default dataset path: {dataset_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {output_dir}")

        # Load configuration
        config = OmegaConf.load(yaml_path)
        # Add debug output to check config structure
        import json
        logger.info("CONFIG STRUCTURE:")
        logger.info(json.dumps(OmegaConf.to_container(config), indent=2))
        if "model" in config and "peft_config" in config.model:
            logger.info("PEFT CONFIG FOUND:")
            logger.info(json.dumps(OmegaConf.to_container(config.model.peft_config), indent=2))
        else:
            logger.warning("No peft_config found in model section of YAML")
        
        # Extract model_name_or_path from config if not provided
        if model_name is None and "variables" in config and "model_name_or_path" in config["variables"]:
            model_name = config["variables"]["model_name_or_path"]
            logger.info(f"Using model name from YAML config: {model_name}")
        elif model_name is None:
            model_name = "meta-llama/Llama-3.2-1B"
            logger.info(f"Using default model name: {model_name}")
        
        # Set HuggingFace token
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token is None:
                raise ValueError("HuggingFace token not found. Please set HF_TOKEN environment variable or pass it as an argument.")
        
        # Set token in environment for transformers
        os.environ["HF_TOKEN"] = hf_token
        
        # Update dataset path in config
        if "train_loader" in config and "dataset" in config["train_loader"]:
            config["train_loader"]["dataset"]["local"] = dataset_path
            logger.info(f"Updated dataset path in config to: {dataset_path}")
        
        # Update eval_loader dataset path if it exists
        if "eval_loader" in config and "dataset" in config["eval_loader"]:
            config["eval_loader"]["dataset"]["local"] = dataset_path
            logger.info(f"Updated eval dataset path in config to: {dataset_path}")

        # Update model configuration - now using the root level model config
        if "model" in config:
            config.model.pretrained_model_name_or_path = model_name
            logger.info(f"Updated model name in config to: {model_name}")
        
        # Set the save folder to the output directory
        config.save_folder = output_dir
        logger.info(f"Set save_folder in config to: {output_dir}")
        
        # Ensure the save folder exists
        os.makedirs(output_dir, exist_ok=True)

        # # Start training
        # logger.info("Starting training")
        # trainer = train(config)
        # logger.info("Training completed successfully")
        
        # THIS IS THE CRITICAL LINE: Register the custom model
        from llmfoundry.models.llama.register import register_custom_llama_model
        register_custom_llama_model()
        logger.info("Registered CustomLlamaModel with registry")

        # Start training
        logger.info("Starting training")
        trainer = train(config)
        logger.info("Training completed successfully")

        return trainer

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a custom Llama model")
    parser.add_argument("--model_name", type=str, default=None,
                      help="Name or path of the pretrained model")
    parser.add_argument("--yaml_path", type=str, default=None,
                      help="Path to the training configuration YAML file")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Directory to save the trained model")
    parser.add_argument("--hf_token", type=str, default=None,
                      help="HuggingFace API token for accessing gated models")
    parser.add_argument("--dataset_path", type=str, default=None,
                      help="Path to the dataset directory")
    
    args = parser.parse_args()
    train_custom_llama(
        model_name=args.model_name,
        yaml_path=args.yaml_path,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        dataset_path=args.dataset_path,
    ) 