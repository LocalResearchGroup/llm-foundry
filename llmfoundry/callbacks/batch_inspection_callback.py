#!/usr/bin/env python3
"""Callback for inspecting batch data passed to the model's forward method."""

import logging
import torch
from typing import Any, Dict
from composer.core import Callback, State
from composer.loggers import Logger

logger = logging.getLogger(__name__)

class BatchInspectionCallback(Callback):
    """Callback that logs detailed information about batches passed to the model."""
    
    def __init__(
        self,
        log_frequency: int = 10,
        sample_size: int = 3,
        log_to_console: bool = True,
        log_to_wandb: bool = True,
    ):
        """Initialize the callback.
        
        Args:
            log_frequency: Log batch info every N batches
            sample_size: Number of sample values to show from each tensor
            log_to_console: Whether to print to console
            log_to_wandb: Whether to log to wandb
        """
        self.log_frequency = log_frequency
        self.sample_size = sample_size
        self.log_to_console = log_to_console
        self.log_to_wandb = log_to_wandb
        self.batch_count = 0
        self.original_forward = None
        
    def _inspect_tensor(self, tensor: torch.Tensor, name: str) -> dict[str, Any]:
        """Extract detailed information from a tensor."""
        info = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad,
        }
        
        # Get sample values
        if tensor.numel() > 0:
            flattened = tensor.flatten()
            
            # For input_ids and labels, show first 10 and last 10
            if name in ['input_ids', 'labels']:
                first_10 = flattened[:10].tolist() if flattened.numel() >= 10 else flattened.tolist()
                last_10 = flattened[-10:].tolist() if flattened.numel() >= 10 else []
                info['first_10'] = first_10
                info['last_10'] = last_10
            else:
                # For other tensors, use original sampling method
                sample_indices = torch.linspace(0, flattened.numel()-1, min(self.sample_size, flattened.numel()), dtype=torch.long)
                samples = flattened[sample_indices].tolist()
                info['samples'] = samples
            
            # Basic statistics for numeric tensors
            if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.int32, torch.int64]:
                info['min'] = tensor.min().item()
                info['max'] = tensor.max().item()
                info['mean'] = tensor.float().mean().item()
        
        return info
    
    def _inspect_batch(self, batch: Any, prefix: str = "") -> dict[str, Any]:
        """Recursively inspect batch structure."""
        batch_info = {}
        
        if isinstance(batch, torch.Tensor):
            return self._inspect_tensor(batch, prefix)
        elif isinstance(batch, dict):
            batch_info['type'] = 'dict'
            batch_info['keys'] = list(batch.keys())
            batch_info['contents'] = {}
            for key, value in batch.items():
                batch_info['contents'][key] = self._inspect_batch(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(batch, (list, tuple)):
            batch_info['type'] = type(batch).__name__
            batch_info['length'] = len(batch)
            batch_info['contents'] = {}
            for i, item in enumerate(batch):
                batch_info['contents'][f'item_{i}'] = self._inspect_batch(item, f"{prefix}[{i}]" if prefix else f"item_{i}")
        else:
            batch_info = {
                'type': type(batch).__name__,
                'value': str(batch)[:100],  # Truncate long strings
            }
        
        return batch_info
    
    def _log_batch_info(self, batch_info: dict[str, Any], state: State, logger: Logger):
        """Log batch information to console and wandb."""
        if self.log_to_console:
            print(f"\n{'='*80}")
            print(f"BATCH INSPECTION - Step {state.timestamp.batch}")
            print(f"{'='*80}")
            self._print_batch_info(batch_info)
            print(f"{'='*80}\n")
        
        if self.log_to_wandb:
            # Flatten the batch info for wandb logging
            flat_info = self._flatten_batch_info(batch_info)
            wandb_metrics = {}
            for key, value in flat_info.items():
                if isinstance(value, (int, float, str, bool)):
                    wandb_metrics[f"batch_inspection/{key}"] = value
            
            if wandb_metrics and hasattr(logger, 'log_metrics'):
                logger.log_metrics(wandb_metrics)
    
    def _print_batch_info(self, info: Any, indent: int = 0):
        """Recursively print batch information."""
        prefix = "  " * indent
        
        if isinstance(info, dict):
            if 'type' in info and info['type'] in ['dict', 'list', 'tuple']:
                print(f"{prefix}Type: {info['type']}")
                if 'keys' in info:
                    print(f"{prefix}Keys: {info['keys']}")
                if 'length' in info:
                    print(f"{prefix}Length: {info['length']}")
                if 'contents' in info:
                    for key, value in info['contents'].items():
                        print(f"{prefix}{key}:")
                        self._print_batch_info(value, indent + 1)
            else:
                # Tensor info
                for key, value in info.items():
                    if key in ['samples', 'first_10', 'last_10']:
                        print(f"{prefix}{key}: {value}")
                    else:
                        print(f"{prefix}{key}: {value}")
        else:
            print(f"{prefix}{info}")
    
    def _flatten_batch_info(self, info: Any, prefix: str = "") -> dict[str, Any]:
        """Flatten nested batch info for wandb logging."""
        flat = {}
        
        if isinstance(info, dict):
            if 'shape' in info:  # Tensor info
                flat[f"{prefix}_shape"] = str(info['shape'])
                flat[f"{prefix}_dtype"] = info['dtype']
                flat[f"{prefix}_device"] = info['device']
                if 'min' in info:
                    flat[f"{prefix}_min"] = info['min']
                if 'max' in info:
                    flat[f"{prefix}_max"] = info['max']
                if 'mean' in info:
                    flat[f"{prefix}_mean"] = info['mean']
            elif 'contents' in info:
                for key, value in info['contents'].items():
                    new_prefix = f"{prefix}_{key}" if prefix else key
                    flat.update(self._flatten_batch_info(value, new_prefix))
        
        return flat
    
    def _create_forward_wrapper(self, original_forward):
        """Create a wrapper around the model's forward method."""
        def forward_wrapper(*args, **kwargs):
            # Inspect the batch (typically the first argument)
            if args and self.batch_count % self.log_frequency == 0:
                batch = args[0] if len(args) > 0 else kwargs
                batch_info = self._inspect_batch(batch)
                
                # We'll store this info to log it in the next callback event
                if hasattr(self, '_current_state') and hasattr(self, '_current_logger'):
                    self._log_batch_info(batch_info, self._current_state, self._current_logger)
            
            self.batch_count += 1
            return original_forward(*args, **kwargs)
        
        return forward_wrapper
    
    def batch_start(self, state: State, logger: Logger) -> None:
        """Called at the start of each batch."""
        # Store state and logger for use in forward wrapper
        self._current_state = state
        self._current_logger = logger
        
        # Wrap the model's forward method if not already wrapped
        if self.original_forward is None:
            model = state.model
            if hasattr(model, 'forward'):
                self.original_forward = model.forward
                model.forward = self._create_forward_wrapper(self.original_forward)
    
    def batch_end(self, state: State, logger: Logger) -> None:
        """Called at the end of each batch."""
        # Clean up references
        self._current_state = None
        self._current_logger = None
    
    def fit_end(self, state: State, logger: Logger) -> None:
        """Restore original forward method when training ends."""
        if self.original_forward is not None:
            state.model.forward = self.original_forward
            self.original_forward = None
