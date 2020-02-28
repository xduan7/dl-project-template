"""
File Name:          debug_wrapper.py
Project:            dl-project-template

File Description:

    Function wrapper (decorator) for debugging purposes. Check the function
    docstring for more details.

    Reference: https://stackoverflow.com/a/39643469

"""
import time
import logging
from typing import Callable, Any
from functools import wraps, partial

# _LOGGER = logging.getLogger(__name__)
_DEBUG_WRAPPER_LOG_LEVEL = logging.INFO


def debug_wrapper(
        func: Callable,
) -> Callable:
    """function wrapper (decorator) for debugging purpose

    This function will log the followings:
    * function entry and exit
    * function execution time
    * function exception if exist
    The log level is configurable with hidden constant
    '_DEBUG_WRAPPER_LOG_LEVEL' in this file.

    :param func: function to be wrapped and debugged
    :return: wrapped function ready for debugging
    """

    @wraps(func)
    def _debug_wrapper(
            *args,
            **kwargs,
    ) -> Any:

        # get the function name, logger, and logging function
        _func_name = func.__name__
        _func_logger = logging.getLogger(_func_name)
        _log = partial(_func_logger.log, level=_DEBUG_WRAPPER_LOG_LEVEL)
        _log(msg=f'Calling function f{_func_name} ... ')

        # fetch and log function params (both args and kwargs)
        _log(msg=f'Function arguments: ')
        _arg_dict = {**dict(zip(func.__code__.co_varnames, args)), **kwargs}
        for _arg_name, _arg_value in _arg_dict.items():
            _log(msg=f'\t\'{_arg_name}\': {_arg_value} ({type(_arg_value)})')

        # execute function and print out exception if necessary
        _ret = None
        _start_time = time.time()
        try:
            _ret = func(*args, **kwargs)
            _log(msg=f'Function f{_func_name} finished properly.')
        except Exception as _exception:
            _exception_str = str(_exception)
            _func_logger.exception(_exception_str)
        _exe_time = time.time() - _start_time

        _log(msg=f'Function execution took {_exe_time:.2f} seconds.')
        return _ret

    return _debug_wrapper
