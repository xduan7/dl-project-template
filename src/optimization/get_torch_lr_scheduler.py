"""
File Name:          get_torch_lr_scheduler.py
Project:            dl-project-template

File Description:

    This file implements a function named 'get_pytorch_lr_scheduler',
    which returns any PyTorch learning rate scheduler with given parameters.

"""
from typing import Dict, Any, Type

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from src.utilities.get_class_from_module import get_class_from_module


def get_pytorch_lr_scheduler(
        lr_scheduler: str,
        optimizer: Optimizer,
        lr_scheduler_kwargs: Dict[str, Any],
) -> LRScheduler:
    """get a PyTorch learning rate scheduler with a given optimizer and
    parameters

    :param lr_scheduler: case-sensitive string for learning rate scheduler
    name
    :param optimizer: PyTorch optimizer for learning rate scheduling
    :param lr_scheduler_kwargs: dictionary keyword arguments for scheduler
    hyper-parameters
    :return: learning rate scheduler of type LRScheduler
    """
    _lr_scheduler_class: Type[LRScheduler] = \
        get_class_from_module(lr_scheduler, torch.optim.lr_scheduler)
    return _lr_scheduler_class(
        optimizer=optimizer,
        **lr_scheduler_kwargs if lr_scheduler_kwargs else {},
    )
