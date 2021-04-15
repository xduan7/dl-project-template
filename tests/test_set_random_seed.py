import random
import unittest
from typing import Tuple

import torch
import numpy as np

from src.utilities import set_random_seed


_RANDOM_SEED: int = random.randint(0, 100)
_TEST_ARRAY_SIZE: Tuple[int, int] = (2, 2)
_TEST_TENSOR_SIZE: Tuple[int, int] = (2, 2)


def _set_random_seed():
    set_random_seed(
        random_seed=_RANDOM_SEED,
    )


class TestSetRandomSeed(unittest.TestCase):
    """Unit test class for ``set_random_seed`` function.

    The test checks the random seed function for Python random,
    NumPy, and PyTorch by asserting the first random number, array,
    or tensor is always the same after seeding.

    """
    def test_random(self):
        _set_random_seed()
        _random = random.random()
        _set_random_seed()
        assert _random == random.random()

    def test_numpy(self):
        _set_random_seed()
        _array = np.random.random(size=_TEST_ARRAY_SIZE)
        _set_random_seed()
        assert (_array == np.random.random(size=_TEST_ARRAY_SIZE)).all()

    def test_torch(self):
        _set_random_seed()
        _tensor = torch.rand(size=_TEST_TENSOR_SIZE)
        _set_random_seed()
        assert (_tensor == torch.rand(size=_TEST_TENSOR_SIZE)).all()


if __name__ == '__main__':
    unittest.main()
