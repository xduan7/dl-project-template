"""
File Name:          test_plot_lr_scheduler.py
Project:            dl-project-template

File Description:

"""
import unittest
from typing import Iterable

import torch
from torch.optim.optimizer import Optimizer

from src.utilities import plot_lr_scheduler
from src.optimization import get_torch_optimizer, get_torch_lr_scheduler


class TestPlotLRScheduler(unittest.TestCase):
    def test_plot_lr_scheduler(self):

        _param: Iterable[torch.Tensor] = \
            [torch.rand((10, 10), requires_grad=True), ]
        _optimizer: Optimizer = get_torch_optimizer(
            optimizer='SGD',
            parameters=_param,
            optimizer_kwargs={
                'lr': 1e-4,
            },
        )
        _lr_scheduler = get_torch_lr_scheduler(
            lr_scheduler='CosineAnnealingWarmRestarts',
            optimizer=_optimizer,
            lr_scheduler_kwargs={
                'T_0': 16,
                'eta_min': 1e-6,
            },
        )
        plot_lr_scheduler(
            lr_scheduler=_lr_scheduler,
            num_epochs=100,
            fig_path=None,
        )


if __name__ == '__main__':
    unittest.main()
