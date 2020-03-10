"""
File Name:          get_computation_devices.py
Project:            dl-project-template

File Description:

    Get the computation devices by checking the user specification against
    system device availability.

"""
import logging
from typing import Optional, List

import torch
from torch import device as Device
try:
    from GPUtil import getAvailable
except (ImportError, ModuleNotFoundError):
    getAvailable = None


# criteria for GPU availability checking
_MAX_NUM_GPUS = 16          # max number of GPUs available
_MAX_GPU_LOAD = 0.05        # max load to be considered as available
_MAX_GPU_MEM_USED = 0.2     # max memory usage to be considered as available


_LOGGER = logging.getLogger(__name__)


def get_computation_devices(
        preferred_gpu_list: Optional[List[int]],
        multi_gpu_flag: bool,
) -> List[Device]:
    """get the available computation devices (CPU & GPUs)

    Get the computation devices for deep learning experiments with given
    preferred list of GPU and flag for multi-GPU computation.

    :param preferred_gpu_list: preferred list of GPUs represented with
    integers starting from 0. For instances, [0, 2] represents th first and
    the third GPUs. None or empty list of GPUS indicate the usage of CPU
    :param multi_gpu_flag: boolean flag for multi-GPU training and testing
    :return: list of integers representing the available devices (CPU & GPUs)
    """

    # use CPU when GPUs are not preferred or not available
    if (preferred_gpu_list is None) \
            or (len(preferred_gpu_list) == 0) \
            or (not torch.cuda.is_available()):
        return [Device('cpu'), ]

    # else GPUs are preferred and available
    # get all available GPU indexes
    _available_gpu_list: List[int]
    if getAvailable:
        # by default, use GPU utility package with load and memory usage
        # specification so that the 'available' GPUs are actually ready
        # for deep learning runs (https://github.com/anderskm/gputil)
        _available_gpu_list = getAvailable(
            limit=_MAX_NUM_GPUS,
            maxLoad=_MAX_GPU_LOAD,
            maxMemory=_MAX_GPU_MEM_USED,
        )
    else:
        # assume all GPUs are good to use without GPUtil package
        _available_gpu_list = list(range(torch.cuda.device_count()))
        _warning_msg = \
            f'GPUtil (https://github.com/anderskm/gputil) not installed.' \
            f'Assuming all GPUs ({_available_gpu_list}) are available ' \
            f'and ready for training ... '
        _LOGGER.warning(_warning_msg)

    # get the overlap between the preferred and the available GPUs
    _gpus = \
        [_g for _g in _available_gpu_list if _g in preferred_gpu_list]

    # use CPU if there is no preferred GPUs that are available
    if len(_gpus) == 0:
        return [Device('cpu'), ]

    # otherwise return one or all GPUs depending on the multi-GPU flag
    return [Device(f'cuda:{_g}') for _g in _gpus] \
        if multi_gpu_flag else [Device(f'cuda:{_gpus[0]}'), ]
