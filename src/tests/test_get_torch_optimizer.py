"""
File Name:          test_get_torch_optimizer.py
Project:            dl-project-template

File Description:

"""
import torch
import inspect
import unittest
from functools import partial
from typing import Tuple, Iterable, List, Dict, Any

from src.optimization import get_torch_optimizer, Optimizer


_PARAMETER_TENSOR_SIZE: Tuple[int, int] = (32, 1024)
_PARAMETERS: Iterable[torch.Tensor] = \
    [torch.rand(size=_PARAMETER_TENSOR_SIZE), ]


def _optimizer_predicate(_obj: Any) -> bool:
    return inspect.isclass(_obj) and issubclass(_obj, Optimizer)


_EXACT_OPTIMIZERS: List[str] = [
    _name for _name, _class in
    inspect.getmembers(torch.optim, _optimizer_predicate)
]
_FUZZY_OPTIMIZERS: List[str] = ['SGd', ]
_TEST_OPTIMIZERS: List[str] = _EXACT_OPTIMIZERS + _FUZZY_OPTIMIZERS
_OPTIMIZER_KWARGS: Dict[str, Any] = {
    # argument for all optimizers
    'lr': 1e-4,
}


class TestGetTorchOptimizer(unittest.TestCase):
    def test_get_torch_optimizer(self):
        _optimizer_kwargs: Dict[str, Any] = {
            'parameters': _PARAMETERS,
            'optimizer_kwargs': _OPTIMIZER_KWARGS,
        }

        for _optimizer in _TEST_OPTIMIZERS:
            if _optimizer != 'Optimizer':
                assert isinstance(
                    get_torch_optimizer(
                        optimizer=_optimizer,
                        **_optimizer_kwargs,
                    ),
                    Optimizer,
                )


if __name__ == '__main__':
    unittest.main()