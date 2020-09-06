"""
File Name:          create_directory.py
Project:            dl-project-template

File Description:

    This file implements a helper function for c a directory if not
    exist.

"""
import os


def create_directory(
        dir_path: str,
):
    """create a directory at given location if not exist

    :param dir_path: directory location to create
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
