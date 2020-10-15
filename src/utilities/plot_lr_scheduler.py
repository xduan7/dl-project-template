"""
File Name:          plot_lr_scheduler.py
Project:            dl-project-template

File Description:

    This file implements a function that plots the learning rates over epochs
    with a given pytorch lr_scheduler.

"""
import logging
from typing import List, Tuple, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


logging.getLogger('matplotlib').setLevel(logging.WARNING)


def plot_lr_scheduler(
        lr_scheduler: LRScheduler,
        num_epochs: int,
        fig_path: Optional[str] = None,
) -> None:
    """plot the learning rate over the epochs of a given scheduler;
    different parameter groups will be plotted with different lines with
    Seaborn; the image would be saved to a given path, or otherwise shown.

    :param lr_scheduler: PyTorch learning rate scheduler that is already
    associated with an optimizer
    :type lr_scheduler: LRScheduler
    :param num_epochs: maximum number of epochs in the plot
    :type num_epochs: int
    :param fig_path: optional path to the learning rate plot; if given,
    the plot would be saved instead of plt.show()
    :type fig_path: str or None
    :return: None
    """

    _optimizer: Optimizer = lr_scheduler.optimizer
    group_lr_list: List[List[float]] = []
    for _ in range(1, 1 + num_epochs):
        # model training code goes here, which includes optimizer.step()
        group_lr_list.append(
            [_group['lr'] for _group in _optimizer.param_groups])
        lr_scheduler.step()

    # construct a dataframe for seaborn line plot
    _lr_list: List[Tuple[int, str, float]] = []
    for _i, _group_lr in enumerate(group_lr_list):
        _curr_epoch: int = _i + 1
        for _group, _lr in enumerate(_group_lr):
            _lr_list.append((_curr_epoch, f'param_group[{_group}]', _lr))
    _lr_df: pd.DataFrame = pd.DataFrame(
        _lr_list,
        columns=['epoch', 'param_group', 'learning_rate'],
    )

    _lr_plot = sns.lineplot(
        x='epoch',
        y='learning_rate',
        hue='param_group',
        data=_lr_df,
    )
    if fig_path:
        _lr_plot.savefig(fig_path)
    else:
        plt.show()
