"""
File Name:          test_get_torch_lr_scheduler.py
Project:            dl-project-template

File Description:

"""
import torch
import inspect
import unittest
from typing import Tuple, Iterable, List, Dict, Any

from src.optimization import get_torch_optimizer, Optimizer
from src.optimization import get_torch_lr_scheduler, LRScheduler


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
    inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass)
]
_FUZZY_LR_SCHEDULERS: List[str] = ['StepR', ]
_TEST_LR_SCHEDULERS: List[str] = _EXACT_LR_SCHEDULERS + _FUZZY_LR_SCHEDULERS
_LR_SCHEDULER_KWARGS: Dict[str, Any] = {
    # positional argument for StepLR
    'step_size': 10,
}


class TestGetTorchLRScheduler(unittest.TestCase):
    def test_get_torch_lr_scheduler(self):
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
