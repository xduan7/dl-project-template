from typing import Dict, Any

import torch

from src.utilities import get_class_from_module


def get_torch_activation(
        activation_name: str,
        activation_kwargs: Dict[str, Any],
) -> Any:
    """Get a PyTorch activation module by name and then initialize it.

    Args:
        activation_name: A string as the name of target activation module.
        activation_kwargs: A dictionary of keyword arguments for the
            returned activation module.

    Returns:
        A PyTorch activation module instance with a name equal or
        similar to the target name, initialized with the keyword
        arguments.

    """
    _activation_class: type = get_class_from_module(
        activation_name,
        torch.nn.modules.activation
    )
    return _activation_class(**activation_kwargs)
