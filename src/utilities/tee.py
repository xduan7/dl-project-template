"""
File Name:          tee.py
Project:            dl-project-template

File Description:

    This file implements a helper class Tee, which copies the stdout to a
    file while keeping things printed in console at the same time.

"""
import os
import sys


class Tee(object):
    """Tee class for copying stdout to a file.
    """

    def __init__(self, log_path: str, mode: str = 'a'):
        """Initialize a Tee class for copying the stdout to a file with given
        name, while maintaining the old stdout (e.g. printing output over
        terminal, etc.)

        :param log_path: path to the Tee log, the parent folder will be
        created upon initialization
        :type log_path: str
        :param mode: writing mode for the log (e.g. 'a' for appending)
        :type mode: str
        """
        self.__stdout = sys.stdout
        self.__log_name = log_path
        self.__mode = mode
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def __del__(self):
        """restore the old stdout (before the initialization of Tee).
        """
        sys.stdout = self.__stdout

    def write(self, stdout_str: str):
        """Write the stdout strings into the file and the old stdout

        :param stdout_str: stdout strings to log and print
        :type stdout_str: str
        """
        with open(self.__log_name, self.__mode) as file:
            file.write(stdout_str)
        self.__stdout.write(stdout_str)

    def flush(self):
        """Flush the output content to the old stdout.
        """
        try:
            self.__stdout.flush()
        except AttributeError:
            pass

    def get_prev_stdout(self):
        """Get the previous/old stdout before Tee.
        """
        return self.__stdout
