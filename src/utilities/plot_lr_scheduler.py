"""
File Name:          plot_lr_scheduler.py
Project:            dl-project-template

File Description:

    This file implements a function that plots the learning rates over epochs
    with a given pytorch lr_scheduler.

"""
from typing import Optional

from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def plot_lr_scheduler(
        lr_scheduler: LRScheduler,
        num_epochs: int,
        fig_path: Optional[str] = None,
):
    """
    TODO: ...

    :param lr_scheduler:
    :type lr_scheduler:
    :param num_epochs:
    :type num_epochs:
    :param fig_path:
    :type fig_path:
    :return:
    :rtype:
    """
    pass
