""" 
File Name:          set_random_seed.py
Project:            dl-project-template
    
File Description:

    Set random state for all packages.

"""
import torch
import random
import logging
import numpy as np


_logger = logging.getLogger(__name__)


def set_random_seed(
        random_seed: int,
        deterministic_cudnn_flag: bool,
) -> None:

    # effective for both CPU and GPU
    torch.manual_seed(random_seed)

    # for strictly deterministic and therefore reproducible results
    # won't effect anything if CuDNN is not installed
    # https://pytorch.org/docs/stable/notes/randomness.html#cudnn
    if torch.cuda.is_available() and deterministic_cudnn_flag:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            _logger.warning(
                'Failed to configure CuDNN for deterministic computation '
                'and therefore reproducible results.'
            )

    random.seed(random_seed)
    np.random.seed(random_seed)

    # TODO: add random seeding for all other packages here
