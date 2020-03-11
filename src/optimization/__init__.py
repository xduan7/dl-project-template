"""
File Name:          __init__.py
Project:            dl-project-template

File Description:

    This module implements the optimizer-related functions and classes

"""
from .get_torch_optimizer import \
    Optimizer, get_torch_optimizer, is_torch_optimizer_class
from .get_torch_lr_scheduler import \
    LRScheduler, get_torch_lr_scheduler, is_torch_lr_scheduler_class

__all__ = [
    # src.optimization.get_torch_optimizer
    'Optimizer',
    'get_torch_optimizer',
    'is_torch_optimizer_class',
    # src.optimization.get_torch_lr_scheduler
    'LRScheduler',
    'get_torch_lr_scheduler',
    'is_torch_lr_scheduler_class',
]
