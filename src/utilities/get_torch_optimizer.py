from typing import Any, Dict, Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from src.utilities import is_subclass, get_class_from_module


def _is_torch_optimizer_class(
        optimizer_class: type,
) -> bool:
    """Check if the given class is a PyTorch optimizer class.

    """
    return is_subclass(optimizer_class, Optimizer)


def get_torch_optimizer(
        optimizer_name: str,
        optimizer_params: Iterable[Tensor],
        optimizer_kwargs: Dict[str, Any],
) -> Optimizer:
    """Get a PyTorch optimizer by name and then initialize it.

    Args:
        optimizer_name: A string as the name of target optimizer.
        optimizer_params: An iterable structure of all the tensors
            to be optimized by the returned optimizer.
        optimizer_kwargs: A dictionary of keyword arguments for the
            returned optimizer.

    Returns:
        A PyTorch optimizer instance with a name equal or similar to
        the target name, initialized with the parameters to be
        optimized and keyword arguments.

    Raises:
        ValueError: No valid optimizer found in the `torch.optim`
            with the given name.

    """
    try:
        _optimizer_class: type = \
            get_class_from_module(optimizer_name, torch.optim)
    except ValueError:
        _error_msg = \
            f'Cannot find an optimizer with a name equal or similar ' \
            f'to \'{optimizer_name}\' in module \'torch.optim\'.'
        raise ValueError(_error_msg)

    if not _is_torch_optimizer_class(_optimizer_class):
        _error_msg = \
            f'Class \'{_optimizer_class}\' found by the name ' \
            f'\'{optimizer_name}\' is not a valid optimizer class.'
        raise ValueError(_error_msg)

    return _optimizer_class(
        params=optimizer_params,
        **optimizer_kwargs,
    )
