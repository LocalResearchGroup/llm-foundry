import pytest
from unittest.mock import patch, MagicMock
import torch
import numpy as np
from pathlib import Path

from llmfoundry.loggers.aim_logger import AimLogger
from composer.core import State

def test_basic_initialization(temp_repo_dir):
    """Test that logger initializes correctly with minimal parameters."""
    logger = AimLogger(repo=str(temp_repo_dir))
    
    assert logger.repo == str(temp_repo_dir)
    assert logger.experiment_name is None
    assert logger._enabled == True
    assert logger._run is None

def test_full_initialization(temp_repo_dir):
    """Test initialization with all optional parameters."""
    logger = AimLogger(
        repo=str(temp_repo_dir),
        experiment_name="test_exp",
        system_tracking_interval=10,
        log_system_params=False,
        capture_terminal_logs=False,
        rank_zero_only=True,
        entity="test_entity",
        project="test_project",
        upload_on_close=False
    )
    
    assert logger.repo == str(temp_repo_dir)
    assert logger.experiment_name == "test_exp"
    assert logger.system_tracking_interval == 10
    assert logger.log_system_params == False
    assert logger.entity == "test_entity"
    assert logger.project == "test_project"
