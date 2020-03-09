"""
File Name:          get_torch_optimizer.py
Project:            dl-project-template

File Description:

    This file implements a function named 'get_pytorch_optimizer',
    which returns any PyTorch optimizer with given parameters.

"""
from inspect import getfullargspec
from typing import Dict, Iterable, Type, List, Any, Optional

import torch
from torch.optim.optimizer import Optimizer

from src.utilities import get_class_from_module, get_valid_kwargs


def get_torch_optimizer(
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

    _valid_optimizer_kwargs: Dict[str, Any] = \
        get_valid_kwargs(
            func=_optimizer_class.__init__,
            kwargs=optimizer_kwargs,
        )

    return _optimizer_class(
        params=parameters,
        **_valid_optimizer_kwargs,
    )
