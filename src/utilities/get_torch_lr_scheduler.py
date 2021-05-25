from typing import Any, Dict

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from src.utilities import is_subclass, get_class_from_module


def _is_torch_lr_scheduler_class(
        lr_scheduler_class: Any,
) -> bool:
    """Check if the given class is a PyTorch LR scheduler class.

    """
    return is_subclass(lr_scheduler_class, LRScheduler)


def get_torch_lr_scheduler(
        lr_scheduler_name: str,
        lr_scheduler_optim: Optimizer,
        lr_scheduler_kwargs: Dict[str, Any],
) -> LRScheduler:
    """Get a PyTorch learning rate (lr) scheduler by name and then
    initialize it.

    Args:
        lr_scheduler_name: A string as the name of target lr scheduler.
        lr_scheduler_optim: The optimizer whose lr to be scheduled.
        lr_scheduler_kwargs: A dictionary of keyword arguments for the
            returned lr scheduler.

    Returns:
        A PyTorch lr scheduler instance with a name equal or similar to
        the target name, initialized with the optimizer whose lr to be
        scheduled and keyword arguments.

    Raises:
        ValueError: No valid optimizer found in the
            `torch.optim.lr_scheduler` with the given name.

    """
    try:
        _lr_scheduler_class: type = \
            get_class_from_module(lr_scheduler_name, torch.optim.lr_scheduler)
    except ValueError:
        _error_msg = \
            f'Cannot find an learning rate scheduler with a name ' \
            f'equal or similar to \'{lr_scheduler_name}\' in ' \
            f'module \'torch.optim.lr_scheduler\'.'
        raise ValueError(_error_msg)

    if not _is_torch_lr_scheduler_class(_lr_scheduler_class):
        _error_msg = \
            f'Class \'{_lr_scheduler_class}\' found by the name ' \
            f'\'{lr_scheduler_name}\' is not a valid learning rate ' \
            f'scheduler class.'
        raise ValueError(_error_msg)

    return _lr_scheduler_class(
        optimizer=lr_scheduler_optim,
        **lr_scheduler_kwargs,
    )
