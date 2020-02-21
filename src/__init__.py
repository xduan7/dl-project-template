import os
import logging

# import desired configurations
from src.configs.example import config
print(f'using configuration \'{config["experiment_name"]}\' ...')

logging.basicConfig(
    # filename=... (could use config['experiment_name'])
    # format=...
    # datefmt=...
    # style=...
    level=logging.DEBUG,
)

# other constants (persistent throughout all experiments)
RAW_DATA_DIR_PATH = os.path.abspath('../data/raw')
ITERM_DATA_DIR_PATH = os.path.abspath('../data/interm')
PROCESSED_DATA_DIR_PATH = os.path.abspath('../data/processed')
LOG_DIR_PATH = os.path.abspath('../logs')
MODEL_DIR_PATH = os.path.abspath('../models')

# __all__
