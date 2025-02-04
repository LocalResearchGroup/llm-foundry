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
        
        logits = self.lm_head(hidden_states)
        
        # @ Z TODO::
        # Typically I don't want my forward calculating the loss, but have to check in with lm foundary to see how it's done
        # loss = None
        # if labels is not None:
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits

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
            logits = self.forward(input_ids, position_ids=position_ids)
            next_token_logits = logits[:, -1, :]
            
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