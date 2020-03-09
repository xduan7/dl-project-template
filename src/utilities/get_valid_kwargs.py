"""
File Name:          get_valid_kwargs.py
Project:            dl-project-template

File Description:

"""
import logging
from inspect import getfullargspec
from typing import Callable, Optional, Dict, Any, List


_LOGGER = logging.getLogger(__name__)


def get_valid_kwargs(
        func: Callable,
        kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:

    _all_possible_args: List[str] = getfullargspec(func).args
    _invalid_args: List[str] = []
    _valid_kwargs: Dict[str, Any] = {}

    for _arg, _v in kwargs.items():
        if _arg in _all_possible_args:
            _valid_kwargs[_arg] = _v
        else:
            _invalid_args.append(_arg)

    if len(_invalid_args) > 0:
        _info_msg = \
            f'Function {func.__qualname__} does not take the any of the ' \
            f'following argument(s): {_invalid_args}. ' \
            f'Continuing by ignoring these argument(s) ...'
        _LOGGER.info(_info_msg)

    return _valid_kwargs
