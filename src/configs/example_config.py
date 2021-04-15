"""Example configuration file.

This is an example of the configuration file, with two code segments:
    - 'private' config variables with type annotation;
    - a 'config' mapping structure that includes all config variables;
This format or segmentation allows users to specify the types for each
config variables with annotation. And since all the variables are
private (starting with '_'), one can simply import the config without
importing all the variables with:
    >>> from configs.example_config import *
or
    >>> from configs.example_config import config

"""
from types import MappingProxyType


_experiment_name: str = 'example_configuration'
_random_state: int = 0

# Network configurations
_linear_stack_layer_dims: List[int] = [1024, 512, 256]
_linear_stack_activation: str = 'ReLU'
_linear_stack_batch_norm: bool = False

# Optimizer configurations
_optimizer: str = 'SGD'
_optimizer__kwargs: Dict[str, Any] = {
    'lr': 1e-4,
    'momentum': 0.9,
}
_lr_scheduler: str = 'StepLR'
_lr_scheduler__kwargs: Dict[str, Any] = {
    'step_size': 10,
}

# The `config` is a read-only mapping that maps names of each
# configuration to their entity with '_' stripped away in the name so
# that they don't get imported. For example: config =
# {'experiment_name': _experiment_name, ...}
config: MappingProxyType = MappingProxyType({
    variable_name[1:]: variable
    for variable_name, variable in locals().items() if
    variable_name.startswith('_') and not variable_name.startswith('__')
})
