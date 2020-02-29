"""
File Name:          get_pytorch_lr_scheduler.py
Project:            dl-project-template

File Description:

    This file implements a function named 'get_pytorch_lr_scheduler',
    which returns any PyTorch learning rate scheduler with given parameters.

"""
import logging
import inspect
from difflib import get_close_matches
from typing import Dict, Any

import torch


_LOGGER = logging.getLogger(__name__)

# dictionary that maps learning rate scheduler names to their PyTorch class
# SKD is short for scheduler
_SKD_CLASS_DICT: Dict[str, type] = {
    _mem[0]: _mem[1]
    for _mem in inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass)
}


def get_pytorch_lr_scheduler(
        lr_scheduler: str,
        optimizer: torch.optim.Optimizer,
        lr_scheduler_kwargs: Dict[str, Any],
) -> Any:
    """get a PyTorch learning rate scheduler with a given optimizer and
    parameters

    :param lr_scheduler: case-sensitive string for learning rate scheduler
    name
    :param optimizer: PyTorch optimizer for learning rate scheduling
    :param lr_scheduler_kwargs: dictionary keyword arguments for scheduler
    hyper-parameters
    :return: learning rate scheduler of type
    torch.optim.lr_scheduler._LRScheduler
    """

    _lr_scheduler_class: type
    if lr_scheduler in _SKD_CLASS_DICT.keys():
        _lr_scheduler_class = _SKD_CLASS_DICT[lr_scheduler]
    else:
        _close_match_skds = \
            get_close_matches(lr_scheduler, _SKD_CLASS_DICT.keys())
        if len(_close_match_skds) == 1:
            _warning_msg = \
                f'Do you mean to use learning rate scheduler ' \
                f'\'{_close_match_skds[0]}\' instead of ' \
                f'\'{lr_scheduler}\'? Continuing with ' \
                f'\'{_close_match_skds[0]}\' learning rate scheduler ...'
            _LOGGER.warning(_warning_msg)
            _lr_scheduler_class = _SKD_CLASS_DICT[_close_match_skds[0]]
        elif len(_close_match_skds) > 1:
            _error_msg = \
                f'Ambiguous learning rate scheduler \'{lr_scheduler}\'.' \
                f'Please pick from any of these schedulers: ' \
                f'{_close_match_skds}'
            raise AttributeError(_error_msg)
        else:
            _error_msg = \
                f'PyTorch learning rate scheduler module ' \
                f'\'torch.optim.lr_scheduler\' has no learning ' \
                f'rate schedulers with name similar to \'{lr_scheduler}\'.'
            raise AttributeError(_error_msg)

    return _lr_scheduler_class(optimizer=optimizer, **lr_scheduler_kwargs)
