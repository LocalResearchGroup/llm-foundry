# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer.loggers import (
    FileLogger,
    InMemoryLogger,
    MLFlowLogger,
    MosaicMLLogger,
    TensorboardLogger,
    WandBLogger,
)

from llmfoundry.loggers.composer_aim_logger import AimLogger

from llmfoundry.registry import loggers

loggers.register('wandb', func=WandBLogger)
loggers.register('tensorboard', func=TensorboardLogger)
loggers.register('inmemory', func=InMemoryLogger)
loggers.register(
    'in_memory_logger',
    func=InMemoryLogger,
)  # for backwards compatibility
loggers.register('mlflow', func=MLFlowLogger)
loggers.register('mosaicml', func=MosaicMLLogger)
loggers.register('aim', func=AimLogger)
loggers.register('file_logger', func=FileLogger)
