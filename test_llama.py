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
        yaml_path = "/llm-foundry/scripts/train/yamls/llama/llama3-1b-lora.yaml"
        print(f"Loading config from {yaml_path}")
        cfg = OmegaConf.load(yaml_path)
        print("Config loaded successfully")

        # Import our custom Llama implementation
        from llmfoundry.models.llama import LlamaForCausalLM
        # from transformers import AutoTokenizer

        # # Initialize tokenizer
        # print("Loading tokenizer...")
        # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

        # print("Tokenizer loaded")

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

        print("Model conversion successful")
        # # Set model to evaluation mode
        # model.eval()

        # # Define prompts
        # prompts = [
        #     "Explain machine learning to a 10-year old:",
        #     "Write a short poem about artificial intelligence:",
        #     "What are the key benefits of using PyTorch for deep learning?"
        # ]

        # # Generate text for each prompt
        # print("\\nGenerating responses...")
        # for i, prompt in enumerate(prompts):
        #     print(f"\\n[Prompt {i+1}]\\n{prompt}\\n")
            
        #     # Tokenize
        #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
        #     # Generate
        #     with torch.no_grad():
        #         try:
        #             output_ids = model.generate(
        #                 inputs.input_ids,
        #                 max_new_tokens=256,
        #                 temperature=0.7,
        #                 top_p=0.9,
        #                 do_sample=True
        #             )
                    
        #             # Decode and print
        #             generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        #             response = generated_text[len(prompt):].strip()
        #             print(f"[Response]\\n{response}\\n")
        #             print("-" * 50)
        #         except Exception as e:
        #             print(f"Error generating response: {e}")
        #             import traceback
        #             traceback.print_exc()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

@app.function(
    gpu="l4", 
    image=image, 
    timeout=3600, 
    secrets=[Secret.from_name("LRG"), Secret.from_name("huggingface-secret")],
    #env={"HF_TOKEN": "{{secret:huggingface-secret.HUGGINGFACE_TOKEN}}"} # Set as environment variable
)
def run_llama():
    import subprocess
    import os
    # Get token from secret and set environment variable
    # Debug environment variables from the secret
    print("Environment variables starting with HUGGINGFACE:")
    for key in os.environ:
        if "HUGGINGFACE" in key:
            print(f"  {key}")
    
    # Try various possible names for the token
    hf_token = (
        os.environ.get("HUGGINGFACE_TOKEN") or 
        os.environ.get("HUGGINGFACE_API_TOKEN") or
        os.environ.get("HF_TOKEN") or
        # Try to access by the secret name directly
        os.environ.get("huggingface-secret") or
        # Try accessing by fully qualified name
        os.environ.get("huggingface-secret.HUGGINGFACE_TOKEN")
    )
    
    if hf_token:
        # Use found token
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        print("HF token found and set")
    else:
        print("WARNING: No HF token found in environment")
        # Continue with public model instead of failing
    # Ensure we're using the right Python
    python_path = "/opt/conda/envs/llm-foundry/bin/python"
    
    # Use the correct Python interpreter for imports
    import_check = subprocess.run(
        [python_path, "-c", "import flash_attn; print(flash_attn.__version__)"],
        capture_output=True,
        text=True
    )
    print(f"Flash Attention version: {import_check.stdout}")
    
    # Step 0: Test Llama implementation
    print("\nTesting Llama3 implementation...")
    success = build_llama()
    if not success:
        print("Llama loading failed!")
        return "Llama loading failed!"
    else:
        print("Llama loading completed successfully!")


    # Step 1: Prepare data - use C4 
    print("\nPreparing data...")
    os.chdir("/llm-foundry/scripts")
    data_prep_cmd = [
        python_path,
        "data_prep/convert_dataset_hf.py",
        "--dataset", "allenai/c4",
        "--data_subset", "en",
        "--out_root", "/root/c4-data",
        "--splits", "train_small", "val_small",  # Separate arguments, not comma-separated
        "--concat_tokens", "2048",
        "--tokenizer", "meta-llama/Llama-3.2-1B",
        "--eos_text", "<|endoftext|>",
        #"--timeout", "60"  # Increase timeout to 60 seconds

    ]
    result = subprocess.run(data_prep_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Data prep errors:", result.stderr)
    
    # Step 2: Fine-tune model with LoRA
    print("\nFine-tuning Llama-3 with LoRA...")
    # train_cmd = [
    #     "composer",
    #     "train/train.py",
    #     "train/yamls/llama/llama3-1b.yaml",
    #     "max_duration=1ep",
    #     "save_folder=/root/llama3-c4",
    #     "device_eval_batch_size=1",
    #     "device_train_microbatch_size=1",
    #     "global_train_batch_size=4",
    #     "variables.data_local=/root/math-data",
    #     "peft.peft_method=lora",
    #     "peft.r=8",
    #     "peft.target_modules=[q_proj,k_proj,v_proj,o_proj]",
    #     "max_duration=10ba",  # Run for exactly 10 batches (steps)
    #     "eval_interval=0",    # Disable evaluation for quick testing
    # ]
    train_cmd = [
    "composer",
    "train/train.py",
    "/llm-foundry/scripts/train/yamls/llama/llama3-1b-lora.yaml", 
    # "variables.data_local=/root/c4-data",
    # f"environment.huggingface_token={os.environ.get('HUGGINGFACE_TOKEN')}"
    ]
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Training errors:", result.stderr)
    
    # Step 3: Convert model to HuggingFace format
    print("\nConverting model to HuggingFace format...")
    convert_cmd = [
        python_path, "inference/convert_composer_to_hf.py",
        "--composer_path", "/root/llama3-c4/latest-rank0.pt",  
        "--hf_output_path", "/root/llama3-c4-hf",
        "--output_precision", "bf16"
    ]
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Conversion errors:", result.stderr)
    
    # Step 4: Evaluate the model on math problems
    print("\nEvaluating model on math problems...")
    eval_cmd = [
        "composer",
        "eval/eval.py",
        "eval/yamls/hf_eval.yaml",
        "icl_tasks=eval/yamls/tasks/c4.yaml",
        "variables.model_name_or_path=/root/llama3-c4-hf"
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Evaluation errors:", result.stderr)
    
    # Step 5: Generate test responses for math problems
    print("\nGenerating test responses for math problems...")
    generate_cmd = [
        python_path, "inference/hf_generate.py",
        "--name_or_path", "/root/llama3-c4-hf",
        "--max_new_tokens", "256",
        "--temperature", "0.1",
        "--top_p", "0.9",
        "--prompts",
        "Question: If a shirt originally costs $25 and is marked down by 20%, what is the new price of the shirt? Answer:",
        "Question: A train travels at a speed of 60 mph. How far will it travel in 3.5 hours? Answer:"
    ]
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Generation errors:", result.stderr)

    return "Llama training and evaluation completed!"

@app.local_entrypoint()
def main():
    run_llama.remote()