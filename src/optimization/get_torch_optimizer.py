"""
File Name:          get_torch_optimizer.py
Project:            dl-project-template

File Description:

    This file implements a function named 'get_pytorch_optimizer',
    which returns any PyTorch optimizer with given parameters.

"""
from typing import Dict, Iterable, Type, Any, Optional

import torch
from torch.optim.optimizer import Optimizer

from src.utilities.get_class_from_module import get_class_from_module


def get_pytorch_optimizer(
        optimizer: str,
        parameters: Iterable,
        optimizer_kwargs: Optional[Dict[str, Any]],
) -> Optimizer:
    """get a PyTorch optimizers with the given algorithm and parameters

    :param optimizer: case-sensitive string for optimization algorithm
    :param parameters: iterable of tensors to optimize
    :param optimizer_kwargs: dictionary keyword arguments for optimizer
    hyper-parameters
    :return: optimizer instance ready to use
    """
    _optimizer_class: Type[Optimizer] = \
        get_class_from_module(optimizer, torch.optim)
    return _optimizer_class(
        params=parameters,
        **optimizer_kwargs if optimizer_kwargs else {},
    )
