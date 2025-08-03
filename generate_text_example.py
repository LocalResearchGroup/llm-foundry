#!/usr/bin/env python3
"""
Example script showing how to generate text with the custom Llama model.
"""

import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Load the custom Llama model and tokenizer."""
    from llmfoundry.models.llama.model import CustomLlamaModel
    from llmfoundry.models.llama.register import register_custom_llama_model
    
    # Register the custom model
    register_custom_llama_model()
    print("✅ Custom Llama model registered")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True
    )
    
    # Load custom model
    model = CustomLlamaModel(
        pretrained_model_name_or_path=MODEL_NAME,
        tokenizer=tokenizer,
        use_flash_attention_2=True,
        max_position_embeddings=512,  # Adjust based on your needs
        use_cache=False  # Important for memory efficiency
    )
    
    # Move to device
    model = model.to(DEVICE)
    
    print(f"✅ Model loaded on {DEVICE}")
    print(f"Model type: {type(model).__name__}")
    
    return model, tokenizer

def generate_text_method1(model, tokenizer, prompt, max_new_tokens=100):
    """Method 1: Using the custom model's generate method."""
    print("\n=== Method 1: Custom Model Generate ===")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Generate using the custom model's generate method
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

def generate_text_method2(model, tokenizer, prompt, max_new_tokens=100):
    """Method 2: Using HuggingFace generate (if available)."""
    print("\n=== Method 2: HuggingFace Generate ===")
    
    # Check if the underlying model has generate method
    if hasattr(model.model, 'generate'):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        model.eval()
        with torch.no_grad():
            outputs = model.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    else:
        return "HuggingFace generate method not available on this model"

def generate_text_method3(model, tokenizer, prompt, max_new_tokens=100):
    """Method 3: Manual token-by-token generation."""
    print("\n=== Method 3: Manual Generation ===")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)
    
    model.eval()
    generated_ids = input_ids.clone()
    
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Forward pass
            outputs = model.forward({
                'input_ids': generated_ids,
                'attention_mask': attention_mask
            })
            
            # Get logits for the last token
            logits = outputs['logits'][:, -1, :]
            
            # Apply temperature
            logits = logits / 0.7
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((attention_mask.shape[0], 1), device=DEVICE)
                ], dim=1)
            
            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    """Main function to demonstrate text generation."""
    print("Loading custom Llama model...")
    model, tokenizer = load_model()
    
    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time, in a distant galaxy",
        "The benefits of renewable energy include"
    ]
    
    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*50}")
        
        try:
            # Method 1: Custom generate
            result1 = generate_text_method1(model, tokenizer, prompt, max_new_tokens=50)
            print(f"Result 1: {result1}")
            
            # Method 2: HF generate (if available)
            result2 = generate_text_method2(model, tokenizer, prompt, max_new_tokens=50)
            print(f"Result 2: {result2}")
            
            # Method 3: Manual generation
            result3 = generate_text_method3(model, tokenizer, prompt, max_new_tokens=50)
            print(f"Result 3: {result3}")
            
        except Exception as e:
            print(f"Error generating text: {e}")
    
    print("\n✅ Text generation examples completed!")

if __name__ == "__main__":
    main() 