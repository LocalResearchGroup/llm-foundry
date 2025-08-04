#!/usr/bin/env python3
"""
Callback for text generation during training.
"""

import logging
from typing import Optional
from composer.core import Callback, State
from composer.loggers import Logger

logger = logging.getLogger(__name__)

class TextGenerationCallback(Callback):
    """Callback that generates text at specific training events."""
    
    def __init__(
        self,
        prompts: Optional[list[str]] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        log_to_wandb: bool = True
    ):
        """
        Initialize the callback.
        
        Args:
            prompts: List of prompts to generate text from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            log_to_wandb: Whether to log to wandb
        """
        self.prompts = prompts or [
            "The future of artificial intelligence is",
            "In a world where technology",
            "The most important thing to remember is",
            "When I think about machine learning"
        ]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.log_to_wandb = log_to_wandb
        
    def _generate_and_log_text(self, state: State, logger: Logger, event_name: str):
        """Generate text and log it."""
        try:
            model = state.model
            if not hasattr(model, 'generate'):
                logger.warning("Model does not have generate method")
                return
                
            print(f"\n{'='*60}")
            print(f"TEXT GENERATION - {event_name}")
            print(f"{'='*60}")
            
            generated_texts = {}
            
            for i, prompt in enumerate(self.prompts):
                try:
                    input_ids = model.tokenizer.encode(prompt, return_tensors='pt').to(model.device)
                    generated_ids = model.generate(
                        idx=input_ids,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        eos_id=model.tokenizer.eos_token_id
                    )
                    generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    print(f"\nPrompt {i+1}: {prompt}")
                    print(f"Generated: {generated_text}")
                    print("-" * 40)                    
                    generated_texts[f"prompt_{i+1}"] = {
                        "prompt": prompt,
                        "generated": generated_text
                    }
                except Exception as e:
                    print(f"Error generating text for prompt {i+1}: {e}")
                    generated_texts[f"prompt_{i+1}"] = {
                        "prompt": prompt,
                        "error": str(e)
                    }
            
            if self.log_to_wandb and hasattr(logger, 'log_metrics'):
                for key, value in generated_texts.items():
                    if "error" not in value:
                        logger.log_metrics({
                            f"generation/{event_name}/{key}/prompt": str(value["prompt"]),
                            f"generation/{event_name}/{key}/text": str(value["generated"])
                        })
                        print(f"WandB Logged: {event_name} - {key}")
                        print(f"  Prompt: {value['prompt']}")
                        print(f"  Generated: {value['generated']}")
                        print()
            
        except Exception as e:
            print(f"Error in text generation callback: {e}")
    
    def fit_start(self, state: State, logger: Logger) -> None:
        """Generate text before training starts."""
        self._generate_and_log_text(state, logger, "BEFORE_TRAINING")
    
    def eval_start(self, state: State, logger: Logger) -> None:
        """Generate text before evaluation starts."""
        self._generate_and_log_text(state, logger, "BEFORE_EVAL")
    
from llmfoundry.registry import callbacks
callbacks.register('text_generation', func=TextGenerationCallback) 