"""
File Name:          residual_block.py
Project:            dl-project-template

File Description:

    Basic implementation of residual block from paper "Deep Residual
    Learning for Image Recognition" by Kaiming He, et al.

    references:
    * https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02
    -intermediate/deep_residual_network/main.py#L76-L113
    * https://github.com/pytorch/vision/blob/master/torchvision/models
    /resnet.py

"""
from typing import Optional

import torch
import torch.nn as nn


def _conv2d_3x3(
        in_channels: int,
        out_channels: int,
        stride: int,
        padding: int = 1,
) -> nn.Module:
    """basic 3-by-3 convolution layer
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
    )


class ResidualBlock(nn.Module):
    """residual block module from the original ResNet
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
    ):
        """constructor for residual block

        This function implements the original/canonical residual block in
        ResNet, with convolution layers, batch normalization, ReLU and
        down-sampling of residual if necessary.
        Down-sampling will be automatically enabled whenever the given
        stride is not 1, or the numbers of channels of input and output do
        not align.

        :param in_channels: number of channels for input
        :param out_channels: number of channels for output
        :param stride: stride length for the first convolution layer,
        which is also the only place where the change of number of channels
        can happen
        """

        super(ResidualBlock, self).__init__()

        self._res_layers = nn.Sequential(
            _conv2d_3x3(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            _conv2d_3x3(out_channels, out_channels, stride=1),
            nn.BatchNorm2d(out_channels),
        )
        # the original implementation in one of the references uses in-place
        # activation functions. However, it is not advised or at least
        # should be used with great caution
        # reference: https://pytorch.org/docs/stable/notes/autograd.html
        # #in-place-operations-with-autograd
        self._activation = nn.ReLU()

        # requires down-sampling if stride is not 1, or the number of
        # in-channels and out-channels do not align
        self._residual_down_sample: Optional[nn.Module]
        if (stride != 1) or (in_channels != out_channels):
            self._residual_down_sample = nn.Sequential(
                _conv2d_3x3(
                    in_channels,
                    out_channels,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self._residual_down_sample = None

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:

        _residual = self._residual_down_sample(x) if \
            self._residual_down_sample else x
        return self._activation(self._res_layers(x) + _residual)
