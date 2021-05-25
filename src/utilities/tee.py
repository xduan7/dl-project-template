"""This file implements a helper class Tee, which copies a std stream
(stdout or stderr by choice) to a file while keeping things printed
on console at the same time.

"""

import os
import sys


class Tee(object):
    """Initialize a Tee class for copying the stdout/stderr to a file
    with the given path, while maintaining the old stdout/stderr (e.g.
    printing output over terminal, etc.)

    Args:
        stream: The std stream to be copied ('stdout' or 'stderr').
        log_path: The path to the Tee log, the parent folder will be
            created upon initialization.
        mode: The writing mode for the log (e.g. 'a' for appending).
    """

    def __init__(
            self,
            stream: str,
            log_path: str,
            mode: str = 'a',
    ):
        self.stream = stream
        if self.stream == 'stdout':
            self.__std_stream = sys.stdout
            sys.stdout = self
        elif self.stream == 'stderr':
            self.__std_stream = sys.stderr
            sys.stderr = self
        else:
            _error_msg = \
                f'Stream must be either \'stdout\' or stderr\'.'
            raise ValueError(_error_msg)
        self.__mode = mode
        self.__log_name = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def __del__(self):
        """Restore the old stdout (before the initialization of Tee)."""
        if self.stream == 'stdout':
            sys.stdout = self.__std_stream
        else:
            sys.stderr = self.__std_stream

    def write(self, std_stream_str: str):
        """Write the stdout strings into the file and the old stdout."""
        with open(self.__log_name, self.__mode) as file:
            file.write(std_stream_str)
        self.__std_stream.write(std_stream_str)

    def flush(self):
        """Flush the output content to the old stdout."""
        try:
            self.__std_stream.flush()
        except AttributeError:
            pass
