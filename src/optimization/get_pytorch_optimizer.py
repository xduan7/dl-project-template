"""
File Name:          get_pytorch_optimizer.py
Project:            dl-project-template

File Description:

    This file implements a function named 'get_pytorch_optimizer',
    which returns any PyTorch optimizer with given parameters.

"""
import logging
import inspect
from difflib import get_close_matches
from typing import Dict, Iterable, Any

import torch


_LOGGER = logging.getLogger(__name__)

# dictionary that maps optimization algorithm names to their PyTorch class
_OPT_CLASS_DICT: Dict[str, type] = {
    _mem[0]: _mem[1]
    for _mem in inspect.getmembers(torch.optim, inspect.isclass)
}


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

    _optimizer_class: type
    if optimizer in _OPT_CLASS_DICT.keys():
        _optimizer_class = _OPT_CLASS_DICT[optimizer]
    else:
        _close_match_algs = \
            get_close_matches(optimizer, _OPT_CLASS_DICT.keys())
        if len(_close_match_algs) == 1:
            _warning_msg = \
                f'Do you mean to use optimization algorithm ' \
                f'\'{_close_match_algs[0]}\' instead of ' \
                f'\'{optimizer}\'? Continuing with ' \
                f'\'{_close_match_algs[0]}\' optimization ...'
            _LOGGER.warning(_warning_msg)
            _optimizer_class = _OPT_CLASS_DICT[_close_match_algs[0]]
        elif len(_close_match_algs) > 1:
            _error_msg = \
                f'Ambiguous optimizer algorithm \'{optimizer}\'.' \
                f'Please pick from any of these algorithms: ' \
                f'{_close_match_algs}'
            raise AttributeError(_error_msg)
        else:
            _error_msg = \
                f'PyTorch optimizer module \'torch.optim\' has no ' \
                f'optimization algorithms with name similar to \'' \
                f'{optimizer}\'.'
            raise AttributeError(_error_msg)

    return _optimizer_class(params=parameters, **optimizer_kwargs)
