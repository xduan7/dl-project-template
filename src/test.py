"""
File Name:          test.py
Project:            dl-project-template

File Description:

"""
import importlib
from types import MappingProxyType, ModuleType

import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from src.modules.linear_block import LinearBlock
from src.utilities.set_random_seed import set_random_seed
from src.optimization.get_torch_optimizer import get_pytorch_optimizer
from src.optimization.get_torch_lr_scheduler import get_pytorch_lr_scheduler


if __name__ == "__main__":

    # import configurations by specifying the configuration file name
    _config_file_name: str = 'example_config'
    _config_file: ModuleType = \
        importlib.import_module(f'src.configs.{_config_file_name}')
    print(type(_config_file))
    config: MappingProxyType = _config_file.config  # type: ignore

    set_random_seed(config['random_state'], True)

    classifier: nn.Module = nn.Sequential(
        LinearBlock(
            layer_dims=config['linear_block_layer_dims'],
            activation=config['linear_block_activation'],
            batch_norm=config['linear_block_batch_norm'],
        ),
        nn.Linear(config['linear_block_layer_dims'][-1], 1),
        nn.Sigmoid(),
    )
    print(classifier)

    optimizer: Optimizer = get_pytorch_optimizer(
        optimizer=config['optimizer'],
        parameters=classifier.parameters(),
        optimizer_kwargs=config['optimizer__kwargs']
    )

    lr_scheduler: LRScheduler = get_pytorch_lr_scheduler(
        lr_scheduler=config['lr_scheduler'],
        optimizer=optimizer,
        lr_scheduler_kwargs=config['lr_scheduler__kwargs']
    )
