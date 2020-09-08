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
#     'data_processes',
#     'utilities',
# ]
# note that this naming scheme might be misleading as the generic names like
# 'modules' and 'utilities' should not be used without any parent package

# constants (persistent throughout all experiments)
SRC_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR_PATH = os.path.join(SRC_DIR_PATH, '../data/raw')
INTERIM_DATA_DIR_PATH = os.path.join(SRC_DIR_PATH, '../data/interim')
PROCESSED_DATA_DIR_PATH = os.path.join(SRC_DIR_PATH, '../data/processed')
LOG_DIR_PATH = os.path.join(SRC_DIR_PATH, '../logs')
MODEL_DIR_PATH = os.path.join(SRC_DIR_PATH, '../models')


logging.basicConfig(
    # filename=... (could use config['experiment_name'])
    # format=...
    # datefmt=...
    # style=...
    level=logging.DEBUG,
)
