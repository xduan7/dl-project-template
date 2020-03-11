"""
File Name:          __init__.py
Project:            dl-project-template

File Description:

    PyTorch modules (nn.Module) or network structures and related functions
    and classes

"""
from .activation import get_torch_activation
from .linear_block import LinearBlock
from .residual_block import ResidualBlock

__all__ = [
    # src.modules.get_torch_activation
    'get_torch_activation',
    # src.modules.linear_block
    'LinearBlock',
    # src.modules.residual_block
    'ResidualBlock',
]
