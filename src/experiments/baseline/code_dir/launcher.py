"""Launcher code for a single NNI baseline trail."""

from argparse import Namespace

import nni
import torch

from utilities import set_random_seed


_default_data_params = Namespace(
    val_ratio=0.2,
    tst_ratio=0.2,
    batch_size=32,
)

_default_model_params = Namespace(
    precision=16,
)

_default_optimization_params = Namespace(
    optimizer_name='AdamW',
    optimizer_kwargs={
        'lr':      1e-3,
        'amsgrad': True,
    },
    lr_scheduler_name='CosineAnnealing',
    lr_scheduler_kwargs={
        'T_max':   25,
        'eta_min': 1e-5,
    },
    gradient_clip_val=10.0,
    min_epochs=50,
    max_epochs=1000,
    early_stop_patience=50,
)

default_params = Namespace(
    random_seed=0,
    **vars(_default_data_params),
    **vars(_default_model_params),
    **vars(_default_optimization_params),
)


if __name__ == '__main__':

    params = nni.utils.merge_parameter(
        default_params,
        nni.get_next_parameter(),
    )
    set_random_seed(params.random_seed)
