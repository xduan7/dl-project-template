"""
File Name:          example.py
Project:            dl-project-template

File Description:

    This is an example of the configuration file, with two code segments:
    (1) 'private' config variables with type annotation;
    (2) a 'config' dict structure that includes all config variables;

    This format or segmentation allows users to specify the types for each
    config variables with annotation. And since all the variables are
    private (starting with '_'), one can simply import the config without
    importing all the variables with:
        from src.configs.example import *
    or
        from src.configs.example import config

"""
from typing import Optional, List


# experiment name associated with this set of configurations
# this could be used for various bookkeeping purposes
_experiment_name: str = 'example'

# random state and deterministic flag for reproducible results
_random_state: int = 0
_deterministic_cudnn_flag: bool = True

# "preferred" GPUs for computation device specification
# to use CPU only, set to None or empty list []; otherwise, set to a list
# of integers representing preferred GPUs for this experiment
_preferred_gpu_list: Optional[List[int]] = [0, 1]

# flag for using multiple GPUs (nn.DataParallel) for this experiment
_multi_gpu_flag: bool = False


CONFIG = {
    'experiment_name': _experiment_name,

    'random_state': _random_state,
    'deterministic_cudnn_flag': _deterministic_cudnn_flag,

    'preferred_gpu_list': _preferred_gpu_list,
    'multi_gpu_flag': _multi_gpu_flag,
}
