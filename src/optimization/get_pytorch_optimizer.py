"""
File Name:          get_pytorch_optimizer.py
Project:            dl-project-template

File Description:

    This file implements a function named 'get_pytorch_optimizer',
    which returns any PyTorch optimizer with given parameters.

"""
from typing import Dict, Iterable, Any

import torch

from src.utilities.get_class_from_module import get_class_from_module


def get_pytorch_optimizer(
        optimizer: str,
        parameters: Iterable,
        optimizer_kwargs: Dict[str, Any],
) -> torch.optim.Optimizer:
    """get a PyTorch optimizers with the given algorithm and parameters

    :param optimizer: case-sensitive string for optimization algorithm
    :param parameters: iterable of tensors to optimize
    :param optimizer_kwargs: dictionary keyword arguments for optimizer
    hyper-parameters
    :return: optimizer instance ready to use
    """
    _optimizer_class: type = get_class_from_module(optimizer, torch.optim)
    return _optimizer_class(params=parameters, **optimizer_kwargs)
