FROM python:3.11-slim

# Install git
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install aim mosaicml transformers datasets peft huggingface_hub wandb