"""Initialization for whole project.

Initialization that applies to all the experiments in the project,
which includes the following steps:
* select desired configuration
* define constants throughout all experiments (paths, etc.)
* logger level and formatting
* miscellaneous things

"""
import os
import logging

from src.configs.example_config import config


# Constants persistent throughout all experiments are declared here
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
