import logging
import unittest
from typing import Any

from utilities import debug_wrapper


_TEST_LOGGING_LEVEL = logging.INFO
_TEST_LOGGING_LEVEL_NAME = \
    logging.getLevelName(_TEST_LOGGING_LEVEL)

_TEST_ARG1 = 1
_TEST_ARG2 = 1.0
_TEST_KWARG1 = '1'
_TEST_KWARG2 = '1'


@debug_wrapper(_TEST_LOGGING_LEVEL)
def _test_func(
        arg1: int,
        arg2: float,
        kwarg1: str = _TEST_KWARG1,
        kwarg2: Any = None,
) -> str:
    return f'{arg1}{arg2}{kwarg1}{kwarg2}'


class TestDebugWrapper(unittest.TestCase):
    """Unit test class for ``debug_wrapper`` function wrapper.

    The test checks the log content and the returned values of
    ``_test_func`` to make sure that the ``debug_wrapper`` logs all
    the arguments (and keyword arguments).

    """
    def test_debug_wrapper(self):
        with self.assertLogs(
                logger='_test_func',
                level=_TEST_LOGGING_LEVEL
        ) as _cm:
            _ret = _test_func(_TEST_ARG1, _TEST_ARG2, kwarg2=_TEST_KWARG2)
            _log_header = f'{_TEST_LOGGING_LEVEL_NAME}:_test_func:'
            # execution time not included for it may vary for machines
            _expected_logs = [
                f'{_log_header}Calling function _test_func ... ',
                f'{_log_header}Function arguments: ',
                f'{_log_header}\targ1 (int): {_TEST_ARG1}',
                f'{_log_header}\targ2 (float): {_TEST_ARG2}',
                f'{_log_header}\tkwarg1 (str): {_TEST_KWARG1}',
                f'{_log_header}\tkwarg2 (str): {_TEST_KWARG2}',
                f'{_log_header}Function _test_func finished properly.',
            ]
            _expected_ret = \
                f'{_TEST_ARG1}{_TEST_ARG2}{_TEST_KWARG1}{_TEST_KWARG2}'
            for __l in _expected_logs:
                self.assertIn(__l, _cm.output)
            self.assertEqual(_expected_ret, _ret)


if __name__ == '__main__':
    unittest.main()
