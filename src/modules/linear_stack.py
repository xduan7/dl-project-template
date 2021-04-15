import warnings
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn

from src.modules import get_torch_activation


class LinearStack(nn.Module):
    """A configurable PyTorch module of a stack of linear layers.

    This class implements a PyTorch module made of a stack of linear
    (dense/fully-connected) layers with configurable options for
    batch normalization, activation function, and dropout rate.
    Instances of this class is made up with multiple layers of [linear,
    batch-normalization*, activation, dropout*] where * means
    optional. Warnings will be raised for unrecommended configurations.

    Args:
        layer_dims: A list of integers for the dimensions of the
            stack of layers. For example, [32, 16, 8] indicates two
            linear layers, transforming a tensor from length of 32
            to 16, and then to 8.
        batch_norm: A boolean flag for batch-normalization.
        activation_name: A case-sensitive string as the name of the
            activation function for all linear layers.
        activation_kwargs: A dictionary of keyword arguments for the
            activation function.
        dropout_rate: The dropout rate in range [0, 1]. Dropout rate
            of 0.0 or less means no dropout layers at all
    """

    def __init__(
            self,
            layer_dims: List[int],
            batch_norm: bool = False,
            activation_name: str = 'ReLU',
            activation_kwargs: Optional[Dict[str, Any]] = None,
            dropout_rate: float = 0.0,
    ):
        super(LinearStack, self).__init__()

        if activation_kwargs \
                and ('inplace' in activation_kwargs) \
                and (activation_kwargs['inplace']):
            _warning_msg = \
                f'Inplace activation functions are not recommended in most ' \
                f'cases. Check https://pytorch.org/docs/stable/notes/' \
                f'autograd.html#in-place-operations-with-autograd ' \
                f'for more details.'
            warnings.warn(_warning_msg)

        if (dropout_rate > 0.0) and batch_norm:
            _warning_msg = \
                f'Using batch-normalization with dropout is not ' \
                f'recommended. Check https://arxiv.org/abs/1801.05134 ' \
                f'for more details.'
            warnings.warn(_warning_msg)

        _layers = nn.ModuleList([])
        for _in_dim, _out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            _layers.append(nn.Linear(_in_dim, _out_dim))
            # ordering: batch-normalization -> activation -> dropout
            if batch_norm:
                _layers.append(nn.BatchNorm1d(num_features=_out_dim))
            _layers.append(
                get_torch_activation(
                    activation_name,
                    activation_kwargs if activation_kwargs else {},
                )
            )
            if dropout_rate > 0.0:
                _layers.append(nn.Dropout(dropout_rate))

        self._layers = nn.Sequential(*_layers)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """Process the input tensor `x` through all the layers."""
        return self._layers(x)
