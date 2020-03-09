"""
File Name:          get_valid_kwargs.py
Project:            dl-project-template

File Description:

"""
from inspect import getfullargspec
from typing import Callable, Optional, Dict, Any, List


def get_valid_kwargs(
        func: Callable,
        kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    _args: List[str] = getfullargspec(func).args
    _valid_kwargs = \
        {
            _arg: _v for _arg, _v in kwargs.items()
            if _arg in _args
        } if kwargs else {}

    return _valid_kwargs
