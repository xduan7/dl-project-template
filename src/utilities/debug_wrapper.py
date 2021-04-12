import time
import logging
from typing import Callable, Any
from inspect import getfullargspec
from functools import wraps, partial


def debug_wrapper(logging_level: int):
    """A factory that generates debug wrappers for functions.

    This function implements a factory with logging level as argument
    for function wrappers with debugging purposes. It wraps the target
    function in ``try ... except ...`` block and logs the start and
    finish of the function call, with function arguments and execution
    time in seconds.

    References:
        https://stackoverflow.com/a/39643469
        https://stackoverflow.com/a/5929165

    Args:
        logging_level: The logging level for debugging messages. The
            logger is under the name of function being debugged.

    Returns:
        A debug wrapper for functions.

    Examples:
        >>> @debug_wrapper(logging.INFO)
        ... function_to_debug(args)

    """
    def _debug_wrapper(function: Callable) -> Callable:

        @wraps(function)
        def _function_with_debug_wrapper(*args, **kwargs) -> Any:

            _function_name = function.__name__
            _function_arg_spec = getfullargspec(function)
            _function_arg_names = _function_arg_spec.args
            # kwargs will replace the default kwargs values
            _function_arg_dict = {
                **dict(zip(
                    _function_arg_names,
                    args + _function_arg_spec.defaults)),
                **kwargs,
            }
            _function_logger = logging.getLogger(_function_name)
            _log = partial(_function_logger.log, level=logging_level)

            _log(msg=f'Calling function {_function_name} ... ')
            _log(msg=f'Function arguments: ')
            for __name, __value in _function_arg_dict.items():
                __type = type(__value).__name__
                _log(msg=f'\t{__name} ({__type}): {__value}')

            _function_ret = None
            _start_time = time.time()
            try:
                _function_ret = function(*args, **kwargs)
                _log(msg=f'Function {_function_name} finished properly.')
            except Exception as _exception:
                _exception_str = str(_exception)
                _function_logger.exception(_exception_str)
            _exe_time = time.time() - _start_time
            _log(msg=f'Function {_function_name} took {_exe_time:.2f} seconds.')
            return _function_ret

        return _function_with_debug_wrapper

    return _debug_wrapper
