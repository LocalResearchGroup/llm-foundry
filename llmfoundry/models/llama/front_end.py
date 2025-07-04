from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .config import LlamaConfig
from .model import LlamaModel

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying uses the head weights as the classifier for the token embeddings for both in and out.
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all layers."""
        # Initialize embeddings
        if hasattr(self.model, 'embed_tokens'):
            nn.init.normal_(self.model.embed_tokens.weight, mean=0.0, std=0.041666666666666664)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for weights
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Zero initialization for biases
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        return hidden_states, self.lm_head.weight

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 30,
        temperature: float = 0.0,
    ) -> torch.LongTensor:
        self.eval()
        bsz, seq_len = input_ids.shape
        
        position_ids = repeat(
            torch.arange(seq_len, device=input_ids.device),
            'l -> b l',
            b=bsz
        )
        
        for _ in range(max_new_tokens):
            hidden_states, classifier_weights = self.forward(input_ids, position_ids=position_ids)
            
            # Get logits by computing hidden_states @ classifier_weights.T
            next_token_logits = hidden_states[:, -1] @ classifier_weights.T
            
            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            new_position_ids = position_ids[:, -1:] + 1
            position_ids = torch.cat([position_ids, new_position_ids], dim=1)
        
        return input_ids