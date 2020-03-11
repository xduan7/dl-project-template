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
    """get the valid keyword arguments of a function

    This function will check all the arguments of the given function,
    and then select a subset of the valid keyword argument dictionary from
    the given one.
    Note that this function DOES NOT check if there are missing arguments.
    Please refer to the function body for more details.

    :param func: function to inspect for arguments
    :param kwargs: a set of keyword arguments to select from
    :return: the maximal subset of valid keyword arguments from the given
    one for the function. Return empty dict if the given kwargs is None or
    none of the arguments are found in the given kwargs
    """

    if kwargs is None:
        return {}

    _func_args_spec = getfullargspec(func)
    _all_possible_args: List[str] = _func_args_spec.args

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

    # the following code is meant to check if there is any missing parameters
    # however, there are two reasons for being commented out
    # (1) missing argument checking is out of the scope for the purpose of
    # this function, which is meant to select valid arguments from dict kwargs
    # (2) can't find a elegant way to deal with 'self' argument
    #
    # _required_args: List[str] = \
    #     _all_possible_args[:-len(_func_args_spec.defaults)] \
    #     if _func_args_spec.defaults else _all_possible_args
    # _missing_args: List[str] = \
    #     [_arg for _arg in _required_args if _arg not in _valid_kwargs.keys()]
    # if len(_missing_args) > 0:
    #     _error_msg = \
    #         f'The following required arguments are missing fom the given ' \
    #         f'keyword arguments for function {func.__qualname__}: ' \
    #         f'{_missing_args}.'
    #     _LOGGER.error(_error_msg)

    return _valid_kwargs
