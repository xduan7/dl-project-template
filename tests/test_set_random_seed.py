"""
File Name:          test_set_random_seed.py
Project:            dl-project-template

File Description:

"""
import random
import unittest
from typing import Tuple

import torch
import numpy as np

from src.utilities import set_random_seed


_RANDOM_SEED: int = random.randint(0, 100)
_TEST_TUPLE_SIZE: Tuple[int, int] = (32, 1024)


def _set_random_seed():
    set_random_seed(
        random_seed=_RANDOM_SEED,
        deterministic_cudnn_flag=True,
    )


class TestSetRandomSeed(unittest.TestCase):
    """unittest class for 'set_random_seed' function
    """

    def test_torch_randomness(self):
        """test 'set_random_seed' function for torch
        """
        _set_random_seed()
        _tensor = torch.rand(size=_TEST_TUPLE_SIZE)

        _set_random_seed()
        assert (_tensor == torch.rand(size=_TEST_TUPLE_SIZE)).all()

    def test_randomness(self):
        """test 'set_random_seed' function for random
        """
        _set_random_seed()
        _random = random.random()

        _set_random_seed()
        assert _random == random.random()

    def test_numpy_randomness(self):
        """test 'set_random_seed' function for numpy
        """
        _set_random_seed()
        _array = np.random.random(size=_TEST_TUPLE_SIZE)

        _set_random_seed()
        assert (_array == np.random.random(size=_TEST_TUPLE_SIZE)).all()


if __name__ == '__main__':
    unittest.main()
