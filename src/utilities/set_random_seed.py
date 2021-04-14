import random
import warnings

import torch
import numpy as np


def set_random_seed(
        random_seed: int,
) -> None:
    """Seed the random generators of random, NumPy, and PyTorch.

    Note that this function will not only set the random state of
    all three packages, but also make the CuDNN strictly deterministic
    for reproducible results.

    References:
        - https://pytorch.org/docs/stable/notes/randomness.html#cudnn

    Args:
        random_seed: A seed/state for random generators.

    Returns:
        None

    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        try:
            _cudnn = torch.backends.cudnn
            _cudnn.deterministic = True
            _cudnn.benchmark = False
        except AttributeError:
            _warning_msg = \
                'Failed to configure CuDNN for deterministic ' \
                'computation and reproducible results.'
            warnings.warn(_warning_msg)
