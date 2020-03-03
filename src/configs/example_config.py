"""
File Name:          example_config.py
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
from types import MappingProxyType
from typing import Optional, List, Dict, Any


# experiment name associated with this set of configurations
# this could be used for various bookkeeping purposes
_experiment_name: str = 'example_configuration'

# random state and deterministic flag for reproducible results
_random_state: int = 0
_deterministic_cudnn_flag: bool = True

# "preferred" GPUs for computation device specification
# to use CPU only, set to None or empty list []; otherwise, set to a list
# of integers representing preferred GPUs for this experiment
_preferred_gpu_list: Optional[List[int]] = [0, 1]

# flag for using multiple GPUs (nn.DataParallel) for this experiment
_multi_gpu_flag: bool = False

# network configurations
_linear_block_layer_dims: List[int] = [1024, 512, 256]
_linear_block_activation: str = 'ReLU'
_linear_block_batch_norm: bool = False

# optimizer configurations
_optimizer: str = 'SGD'
_optimizer__kwargs: Dict[str, Any] = {
    'lr': 1e-4,
    'momentum': 0.9,
}
_lr_scheduler: str = 'StepLR'
_lr_scheduler__kwargs: Dict[str, Any] = {
    'step_size': 10,
}

# read-only dictionary that maps names of each configuration to their object
# e.g. 'experiment_name': _experiment_name, 'random_state': _random_state, etc.
# note that the local configuration variable names must start with '_', but
# the underscores are stripped away in the CONFIG dictionary
config: MappingProxyType = MappingProxyType({
    variable_name[1:]: variable
    for variable_name, variable in locals().items() if
    variable_name.startswith('_') and not variable_name.startswith('__')
})
