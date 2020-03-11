"""
File Name:          linear_block.py
Project:            dl-project-template

File Description:

    This file implements a configurable linear/dense/fully-connected block

"""
import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn

from src.modules import get_torch_activation


_LOGGER = logging.getLogger(__name__)


class LinearBlock(nn.Module):
    """configurable linear (dense/fully-connected) block module
    """

    def __init__(
            self,
            layer_dims: List[int],
            batch_norm: bool = False,
            activation: str = 'ReLU',
            activation_kwargs: Optional[Dict[str, Any]] = None,
            dropout_rate: float = 0.0,
    ):
        """constructor for dense block

        This function constructs a linear (dense/fully-connected) neural
        network block with configurable options. The dimensions of layers,
        activation function (type and options), batch-normalization,
        and dropout rate are all specified in the arguments.
        The constructed block is made up with multiple layers of [linear,
        batch-normalization*, activation, dropout*] where * means optional.
        Note that the function throws warnings for unrecommended
        configurations but will proceed nevertheless.

        :param layer_dims: list of integers indicating the dimensions of
        layers. For example, [32, 16, 8] indicates two linear layers,
        transforming a tensor from length of 32 to 16, and then 8
        :param batch_norm: boolean flag for batch-normalization
        :param activation: case-sensitive string for the name of the
        activation function used for all linear layers
        :param activation_kwargs: dictionary keyword arguments for
        activation function
        :param dropout_rate: float in range [0, 1] for dropout rate.
        Dropout rate of 0.0 means no dropout layers at all
        """

        super(LinearBlock, self).__init__()

        # warning if activation functions are in-place
        if activation_kwargs \
                and ('inplace' in activation_kwargs) \
                and (activation_kwargs['inplace']):
            _warning_msg = \
                f'Inplace activation functions are not recommended in most ' \
                f'cases. Check the link: https://pytorch.org/docs/stable/' \
                f'notes/autograd.html#in-place-operations-with-autograd ' \
                f'for more details.'
            _LOGGER.warning(_warning_msg)

        # warning if dropout and normalization are both present
        if (dropout_rate > 0.0) and batch_norm:
            _warning_msg = \
                f'Using batch-normalization with dropout is not ' \
                f'recommended. Check the link: ' \
                f'https://arxiv.org/abs/1801.05134 for more details.'
            _LOGGER.warning(_warning_msg)

        _layer_list = nn.ModuleList([])
        for _in_dim, _out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            _layer_list.append(nn.Linear(_in_dim, _out_dim))

            # ordering: batch-normalization -> activation -> dropout
            if batch_norm:
                _layer_list.append(nn.BatchNorm1d(num_features=_out_dim))
            _layer_list.append(get_torch_activation(
                activation, activation_kwargs))
            if dropout_rate > 0.0:
                _layer_list.append(nn.Dropout(dropout_rate))

        self._layers = nn.Sequential(*_layer_list)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        return self._layers(x)
