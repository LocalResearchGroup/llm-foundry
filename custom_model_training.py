import os
import modal
import sys
from modal import Image, App, Secret, Volume

import pathlib, datetime

PYTHON_PATH = "/opt/conda/envs/llm-foundry/bin/python"

# command line arguments
TRAINING_GPU = os.environ.get("MODAL_GPU", "L4") 
TRAIN_YAML = os.environ.get("TRAIN_YAML", "")
IS_PEFT = os.environ.get("IS_PEFT", "True")
IS_PEFT = IS_PEFT in ("True", "true")

OUTPUT_PRECISION = os.environ.get("OUTPUT_PRECISION", "bf16")

# defaults --- make sure your Modal Volumes are titled accordingly
DATASET_BASE_PATH = "/datasets"
DATASETS_VOLUME = Volume.from_name("lrg-datasets", create_if_missing=True)
DATASETS_VOLUME_MOUNT_PATH = pathlib.Path("/datasets")
MODEL_CHECKPOINT_VOLUME = Volume.from_name("lrg-model-checkpoints", create_if_missing=True)
MODEL_CHECKPOINT_VOLUME_MOUNT_PATH = pathlib.Path("/model-checkpoints")

app = App("custom-llama-training")

# Build image from local Dockerfile
image = Image.from_dockerfile("Dockerfile", gpu='L4')
image = image.add_local_file(TRAIN_YAML, f"/llm-foundry/scripts/train/yamls/pretrain/{TRAIN_YAML}")
# image = image.add_local_file("train.py", "/llm-foundry/llmfoundry/command_utils/train.py")


@app.function(gpu=TRAINING_GPU, image=image, timeout=12*3600, secrets=[Secret.from_name("LRG")],
              volumes={MODEL_CHECKPOINT_VOLUME_MOUNT_PATH: MODEL_CHECKPOINT_VOLUME,
                      DATASETS_VOLUME_MOUNT_PATH: DATASETS_VOLUME},
              max_containers=1)
def _train(yaml_path):
    import os
    import sys
    import logging
    from pathlib import Path

    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    from llmfoundry.models.llama.register import register_custom_llama_model
    from llmfoundry.command_utils.train import train
    from omegaconf import OmegaConf

    # import text_generation_callback  # type: ignore

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    """Main PEFT training function."""
    logger.info("Registering custom SmolLM2-135M model...")
    register_custom_llama_model()
    logger.info("Custom model registered successfully!")
    
    config_path = f"scripts/train/yamls/pretrain/{yaml_path}"
    logger.info(f"Loading configuration from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    save_folder = "lrg-model-checkpoints/hf_smollm2_135m_peft"
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
    
    if IS_PEFT and hasattr(config.model, 'peft_config') and config.model.peft_config:
        peft_config = config.model.peft_config
        logger.info("PEFT Configuration:")
        logger.info(f"  - Type: {peft_config.peft_type}")
        logger.info(f"  - Rank (r): {peft_config.r}")
        logger.info(f"  - Alpha: {peft_config.lora_alpha}")
        logger.info(f"  - Dropout: {peft_config.lora_dropout}")
        logger.info(f"  - Target modules: {peft_config.target_modules}")
        logger.info(f"  - Use RSLora: {peft_config.get('use_rslora', False)}")
        logger.info(f"  - Use DoRA: {peft_config.get('use_dora', False)}")

        logger.info("Starting PEFT training...")
        try:
            trainer = train(config)
            logger.info("PEFT training completed successfully!")
            logger.info(f"PEFT adapters saved to: {save_folder}")
            del trainer
            return "Training completed successfully"
        except Exception as e:
            logger.error(f"PEFT training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    else:
        logger.warning("No PEFT configuration found!")
        logger.info("Starting Full finetuning training...")
        try:
            trainer = train(config)
            logger.info("Full training completed successfully!")
            del trainer
            return "Training completed successfully"
        except Exception as e:
            logger.error(f"Full training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        return

@app.local_entrypoint()
def main():
    _train.remote(yaml_path=TRAIN_YAML)