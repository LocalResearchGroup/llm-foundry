#!/usr/bin/env python3
"""
Simple script to generate text using a custom Llama model.
Extracted from local_llama_training_instruct.py
"""

import torch
import logging
from transformers import AutoTokenizer
from llmfoundry.models.llama.custom_model import CustomLlamaModel

logging.basicConfig(    
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomLlamaGenerator:
    """A class to handle custom Llama model initialization and text generation."""
    
    def __init__(self, base_model_name: str = "HuggingFaceTB/SmolLM2-135M"):
        """
        Initialize the model and tokenizer once.
        
        Args:
            base_model_name: Base model name for tokenizer
        """
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        logger.info("Initializing custom Llama model...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.model = CustomLlamaModel(
                tokenizer=self.tokenizer,
                model_type="smollm2-135m"
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def generate_text(
        self,
        prompt: str = "Hello, how are you?",
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_k: int = 0,
        context_size: int = 8192
    ):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=max_new_tokens,
                    context_size=context_size,
                    temperature=temperature,
                    top_k=top_k,
                    eos_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(outputs.squeeze(0), skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def generate_text_hf(prompt: str, max_new_tokens: int = 20, temperature: float = 0.0, top_k: int = 0, context_size: int = 8192):
    """Generate text using Hugging Face model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    checkpoint = "HuggingFaceTB/SmolLM2-135M"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer_hf = AutoTokenizer.from_pretrained(checkpoint)
    model_hf = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16).to(device)
    inputs = tokenizer_hf(prompt, return_tensors="pt").to(device)
    outputs = model_hf.generate(**inputs, num_beams=1, do_sample=False, max_length=max_new_tokens)
    return tokenizer_hf.decode(outputs[0]) 

def main():
    """Main function to run text generation with model initialized once."""
    
    prompts = [
        "Hello, how are you?",
        "The answer to life, the universe, and everything is",
        "Write a short poem about coding:",
        "Gravity is"
    ]
    
    print("🤖 Custom Llama Model Text Generation")
    print("=" * 50)
    
    print("Initializing model (this may take a moment)...")
    generator = CustomLlamaGenerator()
    print("Model initialized successfully!")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")
        
        response = generator.generate_text(
            prompt=prompt,
        )
        
        if response:
            print(f"Generated: {response}")
        else:
            print("Generation failed!")
        
        print("-" * 30)

if __name__ == "__main__":
    main() 
    