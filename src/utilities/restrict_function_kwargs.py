import warnings
from inspect import getfullargspec
from typing import Any, Callable, Dict, List


def restrict_function_kwargs(
        func: Callable,
        kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Get the valid keyword arguments from a dictionary of keyword
    arguments for a certain function.

    This function will check the argument space and restrict the
    given set of keyword arguments to the ones that are valid for
    the given function. Warning will be raised if there is invalid
    keyword argument.
    Note that this function only RESTRICT the keyword arguments,
    and therefore does not guarantee that the returned arguments are
    sufficient for the function execution.

    Args:
        func: A target function for keyword arguments.
        kwargs: A given dictionary of keyword arguments that may or
            may not apply to the function.

    Returns:
        A valid set of keyword arguments for the function.

    """
    _func_args = getfullargspec(func).args
    _valid_kwargs: Dict[str, Any] = {}
    _invalid_kwarg_keys: List[str] = []
    for __arg_key, __arg_val in kwargs.items():
        if __arg_key in _func_args:
            _valid_kwargs[__arg_key] = __arg_val
        else:
            _invalid_kwarg_keys.append(__arg_key)

    if len(_invalid_kwarg_keys) > 0:
        _warning_msg = \
            f'Function {func.__qualname__} does not accept the ' \
            f'following argument(s): {_invalid_kwarg_keys}. ' \
            f'Continuing without these invalid argument(s) ...'
        warnings.warn(_warning_msg)

    return _valid_kwargs
