from modal import Image, App, Secret,Mount
import torch
import sys
import torch
import os
from omegaconf import OmegaConf
from huggingface_hub import login
import time
app = App("llama3-test")

# Use the same docker image
image = Image.from_dockerfile("Dockerfile")  
# yaml_path = "/llm-foundry/scripts/train/yamls/llama/llama3-1b-lora.yaml"
# print(f"Loading config from {yaml_path}")
# cfg = OmegaConf.load(yaml_path)
# print("Config loaded successfully")
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


image = image.add_local_dir(
    local_path="/home/mainuser/Desktop/llm-foundry/scripts/train/custom_plugin", 
    remote_path="/llm-foundry/scripts/train/"
)

# Make sure __init__.py file exists
image = image.add_local_file(
    local_path="/home/mainuser/Desktop/llm-foundry/llmfoundry/models/__init__.py",
    remote_path="/llm-foundry/llmfoundry/models/__init__.py"
)




def build_llama(cfg):
    """Create the Python script content for testing the Llama model."""
    


    sys.path.append('/llm-foundry')
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print("Logging in to Hugging Face...")
        login(token=hf_token)
    else:
        print("WARNING: No Hugging Face token found, access to gated models may fail.")
    try:

        # Import our custom Llama implementation
        from llmfoundry.models.llama import LlamaForCausalLM


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
        return model

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise ValueError("Invalid Model Matching")



def llama_generate(model,cfg):
    print("Testing initial custom loaded model generation")
    from transformers import AutoTokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.variables.tokenizer_name)

    print("Tokenizer loaded")
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



################
#todo llama adaptor replace

###############


@app.function(
    gpu="a100", #"l4"
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
    hf_token =  os.environ.get("HF_TOKEN") 
      
    
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
    PYTHON_PATH = "/opt/conda/envs/llm-foundry/bin/python"
    
    # Use the correct Python interpreter for imports
    import_check = subprocess.run(
        [PYTHON_PATH, "-c", "import flash_attn; print(flash_attn.__version__)"],
        capture_output=True,
        text=True
    )
    print(f"Flash Attention version: {import_check.stdout}")


    #debug_file_paths(); return
    # ############ Working steps start############
    # # Step 0: Test Llama implementation: load and generate
    # yaml_path = "/llm-foundry/scripts/train/yamls/llama/llama3-1b-lora.yaml"
    # print(f"Loading config from {yaml_path}")
    # cfg = OmegaConf.load(yaml_path)
    # print("Config loaded successfully")
    
    # # Pass config to build_llama function
    # model = build_llama(cfg)
    
    # # Pass config to llama_generate function
    # llama_generate(model, cfg)
    # ############ Working steps end############
    
    # Create entry script
    #entry_script = create_entry_script()

    # Update YAML
    #yaml_path = os.path.join("/llm-foundry/scripts/train/yamls/llama", "llama3-1b-lora.yaml")
    # Import the adapter to register our custom model
    # print("\nImporting custom adapter...")
    # import_cmd = [
    #     PYTHON_PATH, "-c",
    #     "import sys; sys.path.insert(0, '/llm-foundry/llmfoundry/models/llama/'); import composer_llama_adapter; print('Adapter imported successfully')"
    # ]
    # result = subprocess.run(import_cmd, capture_output=True, text=True)
    #print(result.stdout)
    
    # Now run the training with the registered model
    yaml_path = os.path.join("/llm-foundry/scripts/train/yamls/llama", "llama3-1b-lora.yaml")
    #data_path = "/root/c4-data"

    # # Run the adapter script
    # train_cmd = [
    #     PYTHON_PATH,
    #     "/llm-foundry/llmfoundry/models/llama/composer_llama_adapter.py",
    #     yaml_path
    # ]
    # print("\nRunning training with custom adapter...")
    # result = subprocess.run(train_cmd, capture_output=True, text=True)
    # print(result.stdout)
    # if result.stderr:
    #     print("Training errors:", result.stderr)
    # Run using our entry script
    # os.chdir("/llm-foundry/scripts")
    # print("\nRunning training with custom model...")
    # train_cmd = [PYTHON_PATH, entry_script, yaml_path, "/root/c4-data"]
    # result = subprocess.run(train_cmd, capture_output=True, text=True)
    # print(result.stdout)
    
    print(f"TIME: {time.time()}")
    # Step 1: Prepare data - use C4 TODO: Clean up args to load from yaml
    print("\nPreparing data...")
    os.chdir("/llm-foundry/scripts")
    data_prep_cmd = [
        PYTHON_PATH,
        "data_prep/convert_dataset_hf.py",
        "--dataset", "allenai/c4",
        "--data_subset", "en",
        "--out_root", "/root/c4-data",
        "--splits", "train_small", "val_small",  # Separate arguments, not comma-separated
        "--concat_tokens", "2048",
        "--tokenizer", "meta-llama/Llama-3.2-1B",
        #"--eos_text", "<|endoftext|>",
        #"--timeout", "60"  # Increase timeout to 60 seconds

    ]
    result = subprocess.run(data_prep_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Data prep errors:", result.stderr)


    # Step 4: Run training with adapter
    print("\nRunning training with adapter...")
    yaml_path = "/llm-foundry/scripts/train/yamls/llama/llama3-1b-lora.yaml"
    data_path = "/root/c4-data"

    train_cmd = [
        PYTHON_PATH,
        "/llm-foundry/scripts/train/train_with_llama_adapter.py",
        yaml_path,
        data_path
    ]
    print(f"Running command: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Training errors:", result.stderr)



    # Step 3: Convert model to HuggingFace format
    print("\nConverting model to HuggingFace format...")
    convert_cmd = [
        PYTHON_PATH, "inference/convert_composer_to_hf.py",
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
    # eval_cmd = [
    #     "composer",
    #     "eval/eval.py",
    #     "eval/yamls/hf_eval.yaml",
    #     "icl_tasks=eval/yamls/tasks/c4.yaml",
    #     "variables.model_name_or_path=/root/llama3-c4-hf"
    # ]
    # result = subprocess.run(eval_cmd, capture_output=True, text=True)
    # print(result.stdout)
    # if result.stderr:
    #     print("Evaluation errors:", result.stderr)
    
    model_path = "/root/llama3-c4-hf"
    save_path = "/root/eval_results"
    os.makedirs(save_path, exist_ok=True)
    eval_cmd = [
    "composer",
    "eval/eval.py",
    "eval/yamls/hf_eval.yaml",
    "icl_tasks=eval/yamls/copa.yaml",
    f"variables.model_name_or_path={model_path}",
    f"results_path={save_path}",  # Add results_path parameter
    ]
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Evaluation errors:", result.stderr)
    
    # Step 5: Generate test responses for math problems
    print("\nGenerating test responses for math problems...")
    # generate_cmd = [
    #     PYTHON_PATH, "inference/hf_generate.py",
    #     "--name_or_path", "/root/llama3-c4-hf",
    #     "--max_new_tokens", "256",
    #     "--temperature", "0.1",
    #     "--top_p", "0.9",
    #     "--prompts",
    #     "Question: If a shirt originally costs $25 and is marked down by 20%, what is the new price of the shirt? Answer:",
    #     "Question: A train travels at a speed of 60 mph. How far will it travel in 3.5 hours? Answer:"
    # ]
    #     print("\nGenerating test responses...")
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(yaml_path)
    print("Config loaded successfully")

    # Extract tokenizer name
    tokenizer_name = cfg.variables.tokenizer_name
    print(f"Using tokenizer: {tokenizer_name}")

    # Ensure tokenizer is saved alongside the model
    def fix_tokenizer_after_conversion(model_path:str):
        """Ensure proper tokenizer is saved with converted model"""
        import os
        from transformers import AutoTokenizer
        
        original_tokenizer_path = "meta-llama/Llama-3.2-1B"
        #model_path = "/root/llama3-c4-hf"
        
        # Check if tokenizer files exist
        tokenizer_files = ["tokenizer_config.json", "tokenizer.json"]
        missing = [f for f in tokenizer_files if not os.path.exists(os.path.join(model_path, f))]
        
        if missing:
            print(f"Missing tokenizer files: {missing}. Saving original tokenizer...")
            # Load original tokenizer
            tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_path)
            
            # Save to model path, overwriting any existing files
            tokenizer.save_pretrained(model_path)
            print(f"Original tokenizer saved to {model_path}")
        else:
            print("Tokenizer files found. Testing tokenization...")
            
            # Test if the tokenizer works properly
            saved_tokenizer = AutoTokenizer.from_pretrained(model_path)
            original_tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_path)
            
            # Compare tokenizers
            test_text = "The answer to life, the universe, and happiness is"
            saved_tokens = saved_tokenizer.encode(test_text)
            original_tokens = original_tokenizer.encode(test_text)
            
            if saved_tokens != original_tokens:
                print("WARNING: Saved tokenizer produces different tokens than original!")
                print(f"Saved: {saved_tokens[:10]}...")
                print(f"Original: {original_tokens[:10]}...")
                print("Overwriting with original tokenizer...")
                
                # Save original tokenizer
                original_tokenizer.save_pretrained(model_path)
                print("Original tokenizer saved to replace inconsistent one")
        
        return True

    # Add this after your convert_cmd
    fix_tokenizer_after_conversion(model_path)
    # Save tokenizer
    #save_tokenizer(model_path)
    prompts = None
    if prompts is None:
        prompts = [
            "The answer to life, the universe, and happiness is",
            "Here's a quick recipe for baking chocolate chip cookies: Start by",
        ]
    elif isinstance(prompts, str):
        prompts = [prompts]
    

    print("\nGenerating test responses...")
    generate_cmd = [
        PYTHON_PATH, "inference/hf_generate.py",
        "--name_or_path", model_path, #"meta-llama/Llama-3.2-1B", # model_path,
        "--max_new_tokens", "256",
        "--prompts",
        *prompts,
    ]
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Generation errors:", result.stderr)
    print("Generation complete!")

    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Generation errors:", result.stderr)

    return "Llama training and evaluation completed!"

@app.local_entrypoint()
def main():
    run_llama.remote()



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


#    def create_custom_monkey_patch():
#     """Create a script that directly patches the builders registry"""
#     os.makedirs("/llm-foundry/scripts/train/monkey_patch", exist_ok=True)
    
#     with open("/llm-foundry/scripts/train/monkey_patch/__init__.py", "w") as f:
#         f.write("# Package initialization\n")
    
#     with open("/llm-foundry/scripts/train/monkey_patch/patch.py", "w") as f:
#         f.write("""
# import sys
# sys.path.append('/llm-foundry')

# import torch
# from llmfoundry.models.llama import LlamaForCausalLM
# from composer.models import HuggingFaceModel
# from llmfoundry import registry

# # Define our custom builder function
# def build_custom_llama(
#     pretrained_model_name_or_path,
#     tokenizer=None,
#     **kwargs
# ):
#     print("Using CUSTOM LlamaForCausalLM implementation!")
    
#     # Load model with our custom implementation
#     model = LlamaForCausalLM.from_pretrained(
#         pretrained_model_name_or_path, 
#         torch_dtype=torch.bfloat16,
#         **kwargs
#     )
    
#     print(f"Model type: {type(model).__name__}")
    
#     # Wrap with Composer's model class
#     return HuggingFaceModel(
#         model=model,
#         tokenizer=tokenizer
#     )

# # Directly modify the registry - override the existing implementation
# registry.models.register("hf_causal_lm", func=build_custom_llama)
# print("Successfully patched the registry to use our custom model")
# """)
    
#     return "/llm-foundry/scripts/train/monkey_patch/patch.py"



# def patch_model_registry():
#     """Patch the registry in a way that works with composer"""
#     import sys
#     sys.path.append('/llm-foundry')
    
#     # First, import the necessary modules from llm-foundry
#     from llmfoundry import registry
#     from llmfoundry.models.llama import LlamaForCausalLM
#     from composer.models import HuggingFaceModel
#     import torch
    
#     # Define our custom builder function
#     def build_custom_llama(
#         pretrained_model_name_or_path,
#         tokenizer=None,
#         **kwargs
#     ):
#         print("Building model using CUSTOM LlamaForCausalLM implementation!")
        
#         # Remove any parameters that might cause issues
#         if 'import_path' in kwargs:
#             del kwargs['import_path']
            
#         # Load the model using our custom implementation
#         model = LlamaForCausalLM.from_pretrained(
#             pretrained_model_name_or_path, 
#             torch_dtype=torch.bfloat16,
#             **kwargs
#         )
        
#         print(f"Model type: {type(model).__name__}")
        
#         # Wrap with Composer's model class
#         return HuggingFaceModel(
#             model=model,
#             tokenizer=tokenizer
#         )
    
#     # Register our builder directly with llm-foundry's registry system
#     # This avoids modifying files which was causing circular import issues
#     registry.models.register("llama3_1b")(build_custom_llama)
    
#     print("Successfully registered custom_llama model directly")
#     return True

# 


# def create_small_dataset_converter():
#     """Create a custom dataset conversion script with smaller sample sizes"""
#     script_path = "/llm-foundry/scripts/data_prep/tiny_convert.py"
    
#     with open(script_path, "w") as f:
#         f.write("""
# import sys
# import argparse
# import json
# sys.path.append('/llm-foundry')

# # Import the original functions and classes
# from llmfoundry.command_utils.data_prep.convert_dataset_hf import (
#     convert_dataset_hf, CONSTS, TrainSmallConstants, ValSmallConstants
# )

# # Override the constants with much smaller values
# CONSTS['allenai/c4'].splits['train_small'] = TrainSmallConstants(
#     raw_samples=100,
#     truncated_samples=100
# )

# CONSTS['allenai/c4'].splits['val_small'] = ValSmallConstants(
#     raw_samples=10, 
#     truncated_samples=10
# )

# # Define our own version of the function
# def parse_tokenizer_kwargs(tokenizer_kwargs_str):
#     try:
#         return json.loads(tokenizer_kwargs_str)
#     except json.JSONDecodeError:
#         return {}

# if __name__ == '__main__':
#     # Recreate the argument parsing from the original script
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, required=True)
#     parser.add_argument('--data_subset', type=str, default=None)
#     parser.add_argument('--splits', type=str, nargs='+', required=True)
#     parser.add_argument('--out_root', type=str, required=True)
#     parser.add_argument('--compression', type=str, default=None)
#     parser.add_argument('--concat_tokens', type=int, default=None)
#     parser.add_argument('--tokenizer', type=str, default=None)
#     parser.add_argument('--tokenizer_kwargs', type=str, default='{}')
#     parser.add_argument('--bos_text', type=str, default=None)
#     parser.add_argument('--eos_text', type=str, default=None)
#     parser.add_argument('--no_wrap', action='store_true')
#     parser.add_argument('--num_workers', type=int, default=32)
#     args = parser.parse_args()
    
#     # Parse tokenizer kwargs
#     tokenizer_kwargs = parse_tokenizer_kwargs(args.tokenizer_kwargs)
    
#     # Call the function with the parsed arguments
#     convert_dataset_hf(
#         dataset=args.dataset,
#         data_subset=args.data_subset,
#         splits=args.splits,
#         out_root=args.out_root,
#         compression=args.compression,
#         concat_tokens=args.concat_tokens,
#         tokenizer=args.tokenizer,
#         tokenizer_kwargs=tokenizer_kwargs,
#         bos_text=args.bos_text if args.bos_text else '',
#         eos_text=args.eos_text if args.eos_text else '',
#         no_wrap=args.no_wrap,
#         num_workers=args.num_workers,
#     )
# """)

#     return script_path
# def patch_model_registry():
#     """Patch the llm-foundry model registry to include our custom implementation"""
#     import sys
#     sys.path.append('/llm-foundry')
    
#     # Add your custom model to the registry
#     registry_file = "/llm-foundry/llmfoundry/utils/builders.py"
    
#     # First, read the file
#     with open(registry_file, "r") as f:
#         content = f.read()
    
#     # Let's first print the content to debug
#     print(f"Content length: {len(content)}")
#     print("First few lines:")
#     print(content[:500])
    
#     # Try a more general insertion point
#     insertion_point = content.find("def build_")
#     if insertion_point == -1:
#         print("Could not find any builder function in registry")
#         return False
    
#     # Create the registration code with proper imports
#     registration_code = """
# import torch
# from composer.models import HuggingFaceModel
# from llmfoundry.models.llama import LlamaForCausalLM
# from llmfoundry.utils.builders import registry_module, init_device_and_dtype_meta

# @registry_module(registry_name='models')
# def build_llama3_1b(
#     pretrained_model_name_or_path,
#     tokenizer=None,
#     init_context=None,
#     **kwargs
# ):
#     print("Building model using custom LlamaForCausalLM implementation")
#     with init_device_and_dtype_meta(kwargs) if init_context is None else init_context:
#         model = LlamaForCausalLM.from_pretrained(
#             pretrained_model_name_or_path, 
#             torch_dtype=torch.bfloat16,
#             **kwargs
#         )
#         return HuggingFaceModel(model=model, tokenizer=tokenizer)
# """
    
#     # Insert our code
#     modified_content = content[:insertion_point] + registration_code + content[insertion_point:]
    
#     # Write the modified file
#     with open(registry_file, "w") as f:
#         f.write(modified_content)
    
#     print("Successfully patched model registry")
#     return True
# def create_custom_registry():
#     """Create a custom registry file"""
#     os.makedirs("/llm-foundry/scripts/train/custom_registry", exist_ok=True)
    
#     # Create an __init__.py file
#     with open("/llm-foundry/scripts/train/custom_registry/__init__.py", "w") as f:
#         f.write("# Make the directory a package\n")
    
#     # Create our registry file
#     with open("/llm-foundry/scripts/train/custom_registry/models.py", "w") as f:
#         f.write("""
# import torch
# import sys
# sys.path.append('/llm-foundry')

# # Import the registry decorator
# from llmfoundry.utils.registry_utils import registry_module
# from composer.models import HuggingFaceModel
# from llmfoundry.models.llama import LlamaForCausalLM
# from llmfoundry.utils.builders import init_device_and_dtype_meta

# @registry_module(registry_name='models')
# def build_llama3_1b(
#     pretrained_model_name_or_path,
#     tokenizer=None,
#     init_context=None,
#     **kwargs
# ):
#     print("Building model using custom LlamaForCausalLM implementation")
#     with init_device_and_dtype_meta(kwargs) if init_context is None else init_context:
#         model = LlamaForCausalLM.from_pretrained(
#             pretrained_model_name_or_path, 
#             torch_dtype=torch.bfloat16,
#             **kwargs
#         )
#         return HuggingFaceModel(model=model, tokenizer=tokenizer)
# """)
    
#     return "/llm-foundry/scripts/train/custom_registry/models.py"


# def create_synthetic_dataset():
#     """Create a small synthetic dataset to avoid downloading C4"""
#     import torch
#     from datasets import Dataset
    
#     print("\nCreating small synthetic dataset...")
    
#     # Create directory structure
#     os.makedirs("/root/c4-data/train_small", exist_ok=True)
#     os.makedirs("/root/c4-data/val_small", exist_ok=True)
    
#     # Generate synthetic text samples
#     train_texts = [
#         f"This is training sample {i}. It contains text for testing fine-tuning models. "
#         f"The goal is to train a model efficiently and quickly verify the process works."
#         for i in range(100)
#     ]
    
#     val_texts = [
#         f"This is validation sample {i}. It's used to evaluate model performance."
#         for i in range(10)
#     ]
    
#     # Create datasets
#     train_dataset = Dataset.from_dict({"text": train_texts})
#     val_dataset = Dataset.from_dict({"text": val_texts})
    
#     # Create tokenizer function to match expected format
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
#     def tokenize(examples):
#         return tokenizer(
#             examples["text"],
#             padding=False,
#             truncation=False,
#             return_tensors=None,
#         )
    
#     # Tokenize datasets
#     tokenized_train = train_dataset.map(tokenize, batched=True)
#     tokenized_val = val_dataset.map(tokenize, batched=True)
    
#     # Save to disk
#     tokenized_train.save_to_disk("/root/c4-data/train_small")
#     tokenized_val.save_to_disk("/root/c4-data/val_small")
    
#     print(f"Created synthetic dataset with {len(train_texts)} training and {len(val_texts)} validation samples")
#     return True





# ##############
# def check_and_create_custom_plugin():
#     """Check if custom_plugin exists and create it if not"""
#     custom_plugin_dir = "/llm-foundry/scripts/train/custom_plugin"
    
#     # Create directory if it doesn't exist
#     if not os.path.exists(custom_plugin_dir):
#         print(f"Creating directory: {custom_plugin_dir}")
#         os.makedirs(custom_plugin_dir, exist_ok=True)
#     else:
#         print(f"Directory already exists: {custom_plugin_dir}")
    
#     # Create __init__.py
#     init_file = os.path.join(custom_plugin_dir, "__init__.py")
#     if not os.path.exists(init_file):
#         print(f"Creating {init_file}")
#         with open(init_file, "w") as f:
#             f.write("# Custom plugin package\n")
#     else:
#         print(f"{init_file} already exists")
    
#     # Create model_registry.py
#     registry_file = os.path.join(custom_plugin_dir, "model_registry.py")
#     print(f"Creating/updating {registry_file}")
#     with open(registry_file, "w") as f:
#         f.write("""
# import torch
# from composer.models import HuggingFaceModel
# from llmfoundry.models.llama import LlamaForCausalLM
# from llmfoundry.utils.registry_utils import registry_module
# from llmfoundry import registry

# # Define our model builder function
# def build_llama3_1b(
#     pretrained_model_name_or_path,
#     tokenizer=None,
#     **kwargs
# ):
#     print("Building model using CUSTOM LlamaForCausalLM implementation")
    
#     # Remove any parameters that might cause issues
#     if 'import_path' in kwargs:
#         del kwargs['import_path']
        
#     # Load the model using our custom implementation
#     model = LlamaForCausalLM.from_pretrained(
#         pretrained_model_name_or_path, 
#         torch_dtype=torch.bfloat16,
#         **kwargs
#     )
    
#     print(f"Model type: {type(model).__name__}")
    
#     # Wrap with Composer's model class
#     return HuggingFaceModel(
#         model=model,
#         tokenizer=tokenizer
#     )

# # Register with the registry
# registry.models.register("llama3_1b")(build_llama3_1b)
# print("Registered llama3_1b with the registry")
# """)
    
#     return True

# def modify_build_composer_model():
#     """Add automatic loading of our custom plugin to build_composer_model"""
#     builders_file = "/llm-foundry/llmfoundry/utils/builders.py"
    
#     # Backup the file
#     if os.path.exists(builders_file + ".bak"):
#         print(f"Backup already exists: {builders_file}.bak")
#     else:
#         import shutil
#         shutil.copy(builders_file, builders_file + ".bak")
#         print(f"Created backup: {builders_file}.bak")
    
#     # Read the file
#     with open(builders_file, "r") as f:
#         content = f.read()
    
#     # Add our automatic import
#     if "import train.custom_plugin.model_registry" not in content:
#         # Find the imports section 
#         import_section = content.find("import torch")
#         if import_section == -1:
#             print("Could not find import section in builders.py")
#             return False
        
#         # Add our import
#         modified_content = content[:import_section] + """
# # Import our custom model registry
# try:
#     import sys
#     sys.path.append('/llm-foundry/scripts')
#     import train.custom_plugin.model_registry
#     print("Successfully imported custom model registry")
# except Exception as e:
#     print(f"Error importing custom model registry: {e}")

# """ + content[import_section:]
        
#         # Write the modified file
#         with open(builders_file, "w") as f:
#             f.write(modified_content)
        
#         print("Modified builders.py to import custom model registry")
#     else:
#         print("builders.py already imports custom model registry")
    
#     return True



# ###############

    #####################
#     def test_registry():
#         """Create a test script to verify the registry"""
#         os.makedirs("/llm-foundry/scripts/train/test_script", exist_ok=True)
        
#         with open("/llm-foundry/scripts/train/test_script/test_registry.py", "w") as f:
#             f.write("""
# import sys
# sys.path.append('/llm-foundry')
# sys.path.append('/llm-foundry/scripts') 
# import importlib
# print("Importing module...")
# try:
#     mod = importlib.import_module('train.custom_plugin.model_registry')
#     print("Successfully imported module")
# except Exception as e:
#     print(f"Error importing module: {e}")

# from llmfoundry import registry
# print("Available models:", list(registry.models.get_all().keys()))
# print("Looking for llama3_1b...")
# try:
#     builder = registry.models.get("llama3_1b")
#     print(f"Found llama3_1b: {builder}")
# except Exception as e:
#     print(f"Error getting llama3_1b: {e}")
# """)
        
#         return "/llm-foundry/scripts/train/test_script/test_registry.py"
    
#     # Run the test script
#     test_script = test_registry()
#     os.chdir("/llm-foundry/scripts")
#     print("\nTesting registry...")
#     test_result = subprocess.run([PYTHON_PATH, test_script], capture_output=True, text=True)
#     print(test_result.stdout)
#     if test_result.stderr:
#         print(f"Test errors: {test_result.stderr}")


    ####################

    # # Create our monkey patch
    # patch_path = create_custom_monkey_patch()
    
    # # Immediately import it to modify the registry
    # import subprocess
    # subprocess.run(["python", patch_path])
    

        
    # Step 2: Fine-tune model with LoRA
    # print("\nFine-tuning Llama-3 with LoRA...")
    # yaml_path = os.path.join("/llm-foundry/scripts/train/yamls/llama", "llama3-1b-lora.yaml")
    # #check_and_create_custom_plugin()

    # # Modify build_composer_model to automatically import our plugin
    # #modify_build_composer_model()
    # os.chdir("/llm-foundry/scripts") 
    # train_cmd = [
    #     "composer",
    #     #"--import_module", "train.custom_plugin.model_registry",  # Import our module
    #     "train/train.py",
    #     yaml_path,
    #     "variables.data_local=/root/c4-data",
    #     #"model.name=llama3_1b"  # Override to use our registered model
    # ]
    # print(f"Running command: {' '.join(train_cmd)}")
    # result = subprocess.run(train_cmd, capture_output=True, text=True)
    # print(result.stdout)
    # if result.stderr:
    #     print("Training errors:", result.stderr)
    


# def debug_file_paths():
#     """Debug function to inspect container file system"""
#     import os
#     import subprocess
    
#     print("\n================ CONTAINER FILE SYSTEM DEBUGGING ================")
    
#     # Check if key directories exist
#     key_paths = [
#         "/llm-foundry",
#         "/llm-foundry/scripts",
#         "/llm-foundry/scripts/train",
#         "/llm-foundry/scripts/train/yamls",
#         "/llm-foundry/scripts/train/yamls/llama",
#         "/llm-foundry/llmfoundry",
#         "/llm-foundry/llmfoundry/models",
#         "/llm-foundry/llmfoundry/models/llama"
#     ]
    
#     for path in key_paths:
#         print(f"Does {path} exist? {os.path.exists(path)}")
    
#     # Look for the specific file
#     adapter_script = "/llm-foundry/scripts/train/train_with_llama_adapter.py"
#     print(f"\nDoes {adapter_script} exist? {os.path.exists(adapter_script)}")
#     print("HIIIIIIIIIIIIIIII")
#     adapter_script2 = "/llm-foundry/llmfoundry/models/llama/composer_llama_adapter.py"
#     print(f"\nDoes {adapter_script2} exist? {os.path.exists(adapter_script)}")
#     # List script folder content
#     train_dir = "/llm-foundry/scripts/train"
#     if os.path.exists(train_dir):
#         print(f"\nContents of {train_dir}:")
#         try:
#             print(subprocess.check_output(["ls", "-la", train_dir]).decode())
#         except Exception as e:
#             print(f"Error listing directory: {e}")
    
#     # Show mounted files with mount command
#     print("\nMount information:")
#     try:
#         print(subprocess.check_output(["mount"]).decode())
#     except Exception as e:
#         print(f"Error running mount command: {e}")
    
#     # Check Modal-specific environment
#     print("\nModal-specific environment:")
#     modal_vars = [k for k in os.environ if "MODAL" in k]
#     for var in modal_vars:
#         print(f"  {var}={os.environ[var]}")
    
#     print("\n================ END DEBUGGING ================\n")