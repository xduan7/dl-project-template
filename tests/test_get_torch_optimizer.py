"""
File Name:          test_get_torch_optimizer.py
Project:            dl-project-template

File Description:

"""
import inspect
import unittest
from typing import Any, Dict, List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from src.optimization import get_torch_optimizer
from src.optimization.get_torch_optimizer import \
    _is_torch_optimizer_class


_TEST_OPTIMIZER_PARAMS: List[Tensor] = [torch.rand(size=(8, 32))]
_EXACT_OPTIMIZER_NAMES: List[str] = [
    _name for _name, _class in
    inspect.getmembers(torch.optim, _is_torch_optimizer_class)
]
_FUZZY_OPTIMIZER_NAMES: List[str] = [
    'Sgd',
    'ADAM_',
]
_TEST_OPTIMIZER_NAMES: List[str] = \
    _EXACT_OPTIMIZER_NAMES + _FUZZY_OPTIMIZER_NAMES
_TEST_OPTIMIZER_KWARGS: Dict[str, Any] = {
    'lr': 1e-4,
}


class TestGetTorchOptimizer(unittest.TestCase):
    """Unit test class for ``get_torch_optimizer`` function.

    Test the function ``get_torch_optimizer`` with exact names of the
    optimizers and the fuzzy (slightly incorrect) ones, and check if
    the function could fetch the optimizer instances correctly.

    """
    def test_get_torch_optimizer(self):
        for _optimizer_name in _TEST_OPTIMIZER_NAMES:
            assert isinstance(
                get_torch_optimizer(
                    optimizer_name=_optimizer_name,
                    optimizer_params=_TEST_OPTIMIZER_PARAMS,
                    optimizer_kwargs=_TEST_OPTIMIZER_KWARGS,
                ),
                Optimizer,
            )


if __name__ == '__main__':
    unittest.main()
