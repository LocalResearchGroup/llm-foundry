#!/usr/bin/env python3
"""
Simple script to generate text using a custom Llama model.
Extracted from local_llama_training_instruct.py
"""

import os
import torch
import logging
from transformers import AutoTokenizer
from llmfoundry.models.llama.model import CustomLlamaModel

logging.basicConfig(    
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomLlamaGenerator:
    """A class to handle custom Llama model initialization and text generation."""
    
    def __init__(self, base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
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
            logger.info(f"Loading tokenizer from: {self.base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            logger.info("Tokenizer loaded successfully")
            
            logger.info("Creating custom Llama model with random weights...")
            self.model = CustomLlamaModel(
                pretrained_model_name_or_path=self.base_model_name,
                tokenizer=self.tokenizer,
                use_flash_attention_2=True,
                pretrained=True,  # Random weights
                hidden_size=2048,
                num_attention_heads=16,
                num_key_value_heads=4,
                num_hidden_layers=22,
                vocab_size=128256,
                max_position_embeddings=8192
            )
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device).to(torch.bfloat16)
            logger.info(f"Model moved to {self.device} with bf16 precision")
            logger.info("Model initialization completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def generate_text(
        self,
        prompt: str = "Hello, how are you?",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ):
        """
        Generate text using the initialized model.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling vs greedy decoding
            
        Returns:
            Generated text response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call _initialize_model() first.")
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def main():
    """Main function to run text generation with model initialized once."""
    
    prompts = [
        "Hello, how are you?",
        "The answer to life, the universe, and everything is",
        "Write a short poem about coding:",
        "Explain machine learning in simple terms:"
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
            max_new_tokens=50,
            temperature=1.0,
            top_p=0.9,
            do_sample=True
        )
        
        if response:
            print(f"Generated: {response}")
        else:
            print("Generation failed!")
        
        print("-" * 30)

if __name__ == "__main__":
    main() 