"""
File Name:          get_pytorch_activation.py
Project:            dl-project-template

File Description:

    This file implements a function named 'get_pytorch_activation',
    which returns any PyTorch activation function (as torch.nn.Module
    instance) with given parameters.

"""
from typing import Dict, Any, Optional

import torch

from src.utilities.get_class_from_module import get_class_from_module


def get_pytorch_activation(
        activation: str,
        activation_kwargs: Optional[Dict[str, Any]],
) -> Any:
    """get a PyTorch learning rate scheduler with a given optimizer and
    parameters

    :param activation: case-sensitive string for activation function name
    :param activation_kwargs: dictionary keyword arguments for activation
    function
    :return: activation function as an instance of torch.nn.Module
    """
    _activation_class: type = \
        get_class_from_module(activation, torch.nn.modules.activation)
    return _activation_class(**activation_kwargs if activation_kwargs else {})
