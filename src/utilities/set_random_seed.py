"""
File Name:          set_random_seed.py
Project:            dl-project-template

File Description:

    Set random state for all packages for reproducible results.

"""
import random
import logging

import torch
import numpy as np


_LOGGER = logging.getLogger(__name__)


def set_random_seed(
        random_seed: int,
        deterministic_cudnn_flag: bool,
) -> None:
    """seed the random generators for packages

    Set the random seeds for Numpy and PyTorch, and set CuDNN deterministic
    flag if necessary for strictly reproducible results.

    :param random_seed: integer as random seed
    :param deterministic_cudnn_flag: boolean flag for strictly deterministic
    CuDNN computation
    :return: None
    """

    # effective for both CPU and GPU
    torch.manual_seed(random_seed)

    # for strictly deterministic and therefore reproducible results
    # won't effect anything if CuDNN is not installed
    # https://pytorch.org/docs/stable/notes/randomness.html#cudnn
    if torch.cuda.is_available() and deterministic_cudnn_flag:
        try:
            _cudnn = torch.backends.cudnn
            _cudnn.deterministic = True
            _cudnn.benchmark = False
        except Exception:
            _LOGGER.warning(
                'Failed to configure CuDNN for deterministic computation '
                'and therefore reproducible results.'
            )

    random.seed(random_seed)
    np.random.seed(random_seed)

    # add random seeding for all other packages here
