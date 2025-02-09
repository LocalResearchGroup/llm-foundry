import pytest
import tempfile
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
from composer.core import State
from aim.sdk.run import Run

@pytest.fixture
def temp_repo_dir():
    """Create a temporary directory for the AIM repository."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def mock_aim_run():
    """Create a mock AIM Run object with necessary attributes and methods."""
    mock_run = MagicMock(spec=Run)
    mock_run.hash = "test_hash_123"
    mock_run.repo = MagicMock()
    mock_run.repo.path = "test/path"
    mock_run.track = MagicMock()
    mock_run.close = MagicMock()
    return mock_run

@pytest.fixture
def mock_state():
    """Create a mock Composer State object with basic training attributes."""
    state = MagicMock(spec=State)
    
    # Mock dataloader with batch size
    state.dataloader = MagicMock()
    state.dataloader.batch_size = 32
    
    # Mock basic training attributes
    state.max_duration = "100ep"
    state.run_name = "mock_test_run"
    
    # Mock optimizer
    mock_optimizer = MagicMock()
    mock_optimizer.__class__.__name__ = "Adam"
    state.optimizers = [mock_optimizer]
    
    # Mock model
    state.model = MagicMock()
    state.model.__class__.__name__ = "MockTestModel"
    
    return state

@pytest.fixture
def sample_metrics():
    """Create sample metrics of different types for testing."""
    return {
        # Basic scalar metrics
        'loss': 0.123,
        'accuracy': 0.95,
        
        # Torch tensors
        'tensor_scalar': torch.tensor(0.456),
        'tensor_array': torch.randn(2, 3),
        
        # Numpy array
        'numpy_value': np.array(0.789),
        
        # None value (should be skipped)
        'none_metric': None,
        
        # Integer value
        'step_count': 42
    }
