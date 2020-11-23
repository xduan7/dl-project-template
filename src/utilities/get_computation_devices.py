"""
File Name:          get_computation_devices.py
Project:            dl-project-template

File Description:

    Get the computation devices by checking the user specification against
    system device availability.

"""
import os
import logging
from typing import Optional, Union, List

import torch
from torch import device as Device
try:
    from GPUtil import getAvailable
except (ImportError, ModuleNotFoundError):
    getAvailable = None


# criteria for GPU availability checking
_MAX_NUM_GPUS = float('inf')    # max number of GPUs available
_MAX_GPU_LOAD = 0.2             # max load to be considered as available
_MAX_GPU_MEM_USED = 0.2         # max memory usage considered as available

_LOGGER = logging.getLogger(__name__)


def get_computation_devices(
        preferred_gpu_list: Optional[Union[List[int], str]],
        multi_gpu_flag: bool,
) -> List[Device]:
    """get the available computation devices (CPU & GPUs)

    Get the computation devices for deep learning experiments with given
    preferred list of GPU and flag for multi-GPU computation.

    :param preferred_gpu_list: preferred list of GPUs represented with
    integers starting from 0; e.g. [0, 2] represents th first and
    the third GPUs; none or empty list of GPUS indicate the usage of CPU;
    and 'all' indicates all GPUs are preferred
    :type preferred_gpu_list: Optional[Union[List[int], str]]
    :param multi_gpu_flag: boolean flag for multi-GPU training and testing
    :type multi_gpu_flag: bool
    :return: list of available computation devices (CPU & GPUs) visible to
    PyTorch (not hardware indices from PCIe)
    :rtype: List[Device]
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

    # if CUDA_VISIBLE_DEVICES is set as an environment variable, then make
    # sure that the list of available GPUs are visible, and re-index them in
    # the way that PyTorch can access. For example:
    # CUDA_VISIBLE_DEVICES = [4, 5, 6, 7]
    # GPUs not in use = [1, 4, 7]
    # available GPUs = [4, 7]
    # available GPUs for PyTorch = [0, 3]
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        _available_gpu_list_: List[int] = []
        _cuda_visible_devices: List[int] = \
            [int(_i) for _i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        for __available_gpu_id in _available_gpu_list:
            if __available_gpu_id in _cuda_visible_devices:
                _available_gpu_list_.append(
                    _cuda_visible_devices.index(__available_gpu_id))
        _available_gpu_list = _available_gpu_list_

    # double check to make sure that all GPUs are accessible for PyTorch
    _available_gpu_list_: List[int] = []
    for __available_gpu_id in _available_gpu_list:
        try:
            torch.cuda.get_device_properties(__available_gpu_id)
        except Exception:
            _warning_msg = \
                f'CUDA device {__available_gpu_id}, despite in the range ' \
                f'of PyTorch GPU count, is not available.'
            _LOGGER.warning(_warning_msg)
            print(_warning_msg)
        else:
            _available_gpu_list_.append(__available_gpu_id)
    _available_gpu_list = _available_gpu_list_

    # get the overlap between the preferred and the available GPUs
    if isinstance(preferred_gpu_list, str) and preferred_gpu_list == 'all':
        _gpus = _available_gpu_list
    elif isinstance(preferred_gpu_list, list):
        _gpus = list(set(_available_gpu_list).intersection(preferred_gpu_list))
    else:
        _error_msg = \
            f'Unknown parameter {preferred_gpu_list} for the list of ' \
            f'preferred GPUs. Must be either \"all\" or list of integers.'
        raise ValueError(_error_msg)

    # use CPU if there is no preferred GPUs that are available
    if len(_gpus) == 0:
        return [Device('cpu'), ]

    # otherwise return one or all GPUs depending on the multi-GPU flag
    return [Device(f'cuda:{_g}') for _g in _gpus] \
        if multi_gpu_flag else [Device(f'cuda:{_gpus[0]}'), ]
