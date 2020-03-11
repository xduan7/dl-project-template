"""
File Name:          test_get_torch_lr_scheduler.py
Project:            dl-project-template

File Description:

    function for getting PyTorch learning rate scheduler, and its helper
    functions and classes

"""
import inspect
import unittest
from typing import Tuple, Iterable, List, Dict, Any

import torch

from src.optimization import Optimizer, get_torch_optimizer
from src.optimization import \
    LRScheduler, is_torch_lr_scheduler_class, get_torch_lr_scheduler


_PARAMETER_TENSOR_SIZE: Tuple[int, int] = (32, 1024)
_PARAMETERS: Iterable[torch.Tensor] = \
    [torch.rand(size=_PARAMETER_TENSOR_SIZE), ]
_OPTIMIZER: Optimizer = get_torch_optimizer(
    optimizer='SGD',
    parameters=_PARAMETERS,
    optimizer_kwargs={'lr': 1e-4},
)

_EXACT_LR_SCHEDULERS: List[str] = [
    _name for _name, _class in
    inspect.getmembers(torch.optim.lr_scheduler, is_torch_lr_scheduler_class)
]
_FUZZY_LR_SCHEDULERS: List[str] = ['SteplR', ]
_TEST_LR_SCHEDULERS: List[str] = _EXACT_LR_SCHEDULERS + _FUZZY_LR_SCHEDULERS
_LR_SCHEDULER_KWARGS: Dict[str, Any] = {
    # argument for StepLR
    'step_size': 10,
    # arguments for CosineAnnealingLR
    'T_max': 128,
    # arguments for CosineAnnealingWarmRestarts
    'T_0': 8,
    # arguments for CyclicLR
    'base_lr': 1e-4,
    'max_lr': 1e-3,
    # arguments for ExponentialLR
    'gamma': 0.5,
    # arguments for LambdaLR
    'lr_lambda': lambda _lr: _lr * 0.5,
    # arguments for MultiStepLR
    'milestones': [2**_e for _e in range(4)],
    # arguments for OneCycleLR
    # 'max_lr': 1e-2,
    'total_steps': 128,
}


class TestGetTorchLRScheduler(unittest.TestCase):
    """unittest class for 'get_torch_lr_scheduler' function
    """
    def test_get_torch_lr_scheduler(self):
        """test 'get_torch_lr_scheduler' function
        """
        _lr_scheduler_kwargs: Dict[str, Any] = {
            'optimizer': _OPTIMIZER,
            'lr_scheduler_kwargs': _LR_SCHEDULER_KWARGS,
        }

        for _lr_scheduler in _TEST_LR_SCHEDULERS:
            assert isinstance(
                get_torch_lr_scheduler(
                    lr_scheduler=_lr_scheduler,
                    **_lr_scheduler_kwargs,
                ),
                LRScheduler,
            )


if __name__ == '__main__':
    unittest.main()
