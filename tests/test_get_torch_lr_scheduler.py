import inspect
import unittest
from typing import Any, Dict, List

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from src.optimization import get_torch_optimizer
from src.optimization import get_torch_lr_scheduler
from src.optimization.get_torch_lr_scheduler import \
    _is_torch_lr_scheduler_class


_TEST_LR_SCHEDULER_OPTIM: Optimizer = get_torch_optimizer(
    optimizer_name='SGD',
    optimizer_params=[torch.rand(size=(8, 32))],
    optimizer_kwargs={'lr': 1e-4},
)

_EXACT_LR_SCHEDULER_NAMES: List[str] = [
    _name for _name, _class in
    inspect.getmembers(torch.optim.lr_scheduler, _is_torch_lr_scheduler_class)
]
_FUZZY_LR_SCHEDULER_NAMES: List[str] = [
    'SteplR',
]
_TEST_LR_SCHEDULER_NAMES: List[str] = \
    _EXACT_LR_SCHEDULER_NAMES + _FUZZY_LR_SCHEDULER_NAMES
_TEST_LR_SCHEDULER_KWARGS_DICT: Dict[str, Dict[str, Any]] = {
    'CosineAnnealingLR': {
        'T_max': 10,
    },
    'CosineAnnealingWarmRestarts': {
        'T_0': 5,
    },
    'CyclicLR': {
        'base_lr': 1e-4,
        'max_lr': 1e-3,
    },
    'ExponentialLR': {
        'gamma': 0.2,
    },
    'LambdaLR': {
        'lr_lambda': lambda __e: 0.95 ** __e,
    },
    'MultiStepLR': {
        'milestones': [16, 64, 256],
    },
    'MultiplicativeLR': {
        'lr_lambda': lambda __e: 0.95 ** __e,
    },
    'OneCycleLR': {
        'max_lr': 1e-3,
        'total_steps': 100,
    },
    'StepLR': {
        'step_size': 10,
    },
    'SteplR': {
        'step_size': 10,
    }
}


class TestGetTorchLRScheduler(unittest.TestCase):
    """Unit test class for ``get_torch_lr_scheduler`` function.

    Test the function ``get_torch_lr_scheduler`` with exact names of the
    lr scheduler and the fuzzy (slightly incorrect) ones, and check if
    the function could fetch the scheduler instances correctly.

    """
    def test_get_torch_lr_scheduler(self):
        for _lr_scheduler_name in _TEST_LR_SCHEDULER_NAMES:
            _lr_scheduler_kwargs = \
                _TEST_LR_SCHEDULER_KWARGS_DICT.get(_lr_scheduler_name, {})
            assert isinstance(
                get_torch_lr_scheduler(
                    lr_scheduler_name=_lr_scheduler_name,
                    lr_scheduler_optim=_TEST_LR_SCHEDULER_OPTIM,
                    lr_scheduler_kwargs=_lr_scheduler_kwargs,
                ),
                LRScheduler,
            )


if __name__ == '__main__':
    unittest.main()
