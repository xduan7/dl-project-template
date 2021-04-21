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

    Examples:
        >>> @debug_wrapper(logging.INFO)
        ... function_to_debug(args)

    Args:
        logging_level: The logging level for debugging messages. The
            logger is under the name of function being debugged.

    Returns:
        A debug wrapper for functions.

    """
    def _debug_wrapper(func: Callable) -> Callable:

        @wraps(func)
        def _func_with_debug_wrapper(*args, **kwargs) -> Any:

            _func_name = func.__name__
            _func_arg_spec = getfullargspec(func)
            _func_arg_names = _func_arg_spec.args
            # kwargs will replace the default kwargs values
            _func_default_args = \
                (_func_arg_spec.defaults if _func_arg_spec.defaults else ())
            _function_arg_dict = {
                **dict(zip(
                    _func_arg_names,
                    args + _func_default_args)),
                **kwargs,
            }
            _function_logger = logging.getLogger(_func_name)
            _log = partial(_function_logger.log, level=logging_level)

            _log(msg=f'Calling function {_func_name} ... ')
            _log(msg=f'Function arguments: ')
            for __name, __value in _function_arg_dict.items():
                __type = type(__value).__name__
                _log(msg=f'\t{__name} ({__type}): {__value}')

            _func_ret = None
            _start_time = time.time()
            try:
                _func_ret = func(*args, **kwargs)
                _log(msg=f'Function {_func_name} finished properly.')
            except Exception as _exception:
                _exception_str = str(_exception)
                _function_logger.exception(_exception_str)
            _exe_time = time.time() - _start_time
            _log(msg=f'Function {_func_name} took {_exe_time:.2f} seconds.')
            return _func_ret

        return _func_with_debug_wrapper

    return _debug_wrapper
