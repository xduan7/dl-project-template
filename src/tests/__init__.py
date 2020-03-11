"""
File Name:          __init__.py
Project:            dl-project-template

File Description:

    This module implements unit tests

"""
from .test_get_torch_optimizer import TestGetTorchOptimizer
from .test_get_torch_lr_scheduler import TestGetTorchLRScheduler
from .test_set_random_seed import TestSetRandomSeed

__all__ = [
    # src.tests.get_torch_activation
    'TestGetTorchOptimizer',
    # src.tests.test_get_torch_lr_scheduler
    'TestGetTorchLRScheduler',
    # src.tests.test_set_random_seed
    'TestSetRandomSeed',
]
