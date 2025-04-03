from modal import Image, App, Secret,Mount

app = App("llama3-test")

# Use the same docker image
image = Image.from_dockerfile("Dockerfile")

# Mount your local YAML directory
# yaml_mount = Mount.from_local_dir(
#     local_path="/home/mainuser/Desktop/llm-foundry/scripts/train/yamls/llama",  # Full path
#     remote_path="/llm-foundry/scripts/train/yamls/llama"  # Path in container
# )

# Mount both the YAML directory and your llama module code
image = image.add_local_dir(
    local_path="/home/mainuser/Desktop/llm-foundry/scripts/train/yamls/llama", 
    remote_path="/llm-foundry/scripts/train/yamls/llama"
)

# Mount your custom llama module implementation
image = image.add_local_dir(
    local_path="/home/mainuser/Desktop/llm-foundry/llmfoundry/models/llama", 
    remote_path="/llm-foundry/llmfoundry/models/llama"
)

# Make sure __init__.py file exists
image = image.add_local_file(
    local_path="/home/mainuser/Desktop/llm-foundry/llmfoundry/models/__init__.py",
    remote_path="/llm-foundry/llmfoundry/models/__init__.py"
)

def build_llama():
    """Create the Python script content for testing the Llama model."""
    
    import sys
    import torch
    import os
    from omegaconf import OmegaConf
    from huggingface_hub import login

    sys.path.append('/llm-foundry')
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("Logging in to Hugging Face...")
        login(token=hf_token)
    else:
        print("WARNING: No Hugging Face token found, access to gated models may fail.")
    try:
        # Load YAML config - use the existing file
        yaml_path = "/llm-foundry/scripts/train/yamls/llama/llama3-1b.yaml"
        print(f"Loading config from {yaml_path}")
        cfg = OmegaConf.load(yaml_path)
        print("Config loaded successfully")

        # Import our custom Llama implementation
        from llmfoundry.models.llama import LlamaForCausalLM
        from transformers import AutoTokenizer

        # Initialize tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

        # Build model from config
        print("Building model...")
        model_cfg = dict(cfg.model)
        model = LlamaForCausalLM.from_config(model_cfg)
        print(f"Model built successfully - {model.__class__.__name__}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params/1e6:.2f}M parameters")

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Moving model to {device}...")
        model = model.to(device)
        
        # Convert to bfloat16 for Flash Attention
        print("Converting model to bfloat16...")
        model = model.to(torch.bfloat16)


        # Set model to evaluation mode
        model.eval()

        # Define prompts
        prompts = [
            "Explain machine learning to a 10-year old:",
            "Write a short poem about artificial intelligence:",
            "What are the key benefits of using PyTorch for deep learning?"
        ]

        # Generate text for each prompt
        print("\\nGenerating responses...")
        for i, prompt in enumerate(prompts):
            print(f"\\n[Prompt {i+1}]\\n{prompt}\\n")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate
            with torch.no_grad():
                try:
                    output_ids = model.generate(
                        inputs.input_ids,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True
                    )
                    
                    # Decode and print
                    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    response = generated_text[len(prompt):].strip()
                    print(f"[Response]\\n{response}\\n")
                    print("-" * 50)
                except Exception as e:
                    print(f"Error generating response: {e}")
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

@app.function(gpu="l4", image=image, timeout=3600, secrets=[Secret.from_name("LRG"),
                                                            Secret.from_name("huggingface-secret")])
def run_llama():
    import subprocess
    
    # Ensure we're using the right Python
    python_path = "/opt/conda/envs/llm-foundry/bin/python"
    
    # Use the correct Python interpreter for imports
    import_check = subprocess.run(
        [python_path, "-c", "import flash_attn; print(flash_attn.__version__)"],
        capture_output=True,
        text=True
    )
    print(f"Flash Attention version: {import_check.stdout}")
    
    # Test Llama implementation
    print("\nTesting Llama3 implementation...")
    success = build_llama()
    if not success:
        print("Llama testing failed!")
    else:
        print("Llama testing completed successfully!")

    return "Llama test completed!"


@app.local_entrypoint()
def main():
    run_llama.remote()