"""
File Name:          __init__.py
Project:            dl-project-template

File Description:

    Initialization for the project overall, which includes the following
    steps in particular ordering:
    * import necessary modules
    * import desired configuration ('CONFIG') from 'src.configs'
    * define constants throughout all experiments (paths, etc.)
    * logger level and formatting
    * package function and class specification
    * miscellaneous

"""
import os
import logging

# check the following article for organized __init__:
# https://towardsdatascience.com/whats-init-for-me-d70a312da583
# from src import *
# __all__ = [
#     'configs',
#     'modules',
#     'optimization',
#     'processes',
#     'utilities',
# ]
# note that this naming scheme might be misleading as the generic names like
# 'modules' and 'utilities' should not be used without any parent package

# constants (persistent throughout all experiments)
RAW_DATA_DIR_PATH = os.path.abspath('../data/raw')
ITERM_DATA_DIR_PATH = os.path.abspath('../data/interim')
PROCESSED_DATA_DIR_PATH = os.path.abspath('../data/processed')
LOG_DIR_PATH = os.path.abspath('../logs')
MODEL_DIR_PATH = os.path.abspath('../models')


logging.basicConfig(
    # filename=... (could use config['experiment_name'])
    # format=...
    # datefmt=...
    # style=...
    level=logging.DEBUG,
)
