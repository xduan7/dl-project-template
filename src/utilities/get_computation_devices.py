""" 
File Name:          get_computation_devices.py
Project:            dl-project-template
    
File Description:   

    Get the computation devices by checking the user specification against
    system device availability.

"""
import torch
import logging
from typing import Optional, List


# criteria for GPU availability checking
_MAX_NUM_GPUS = 16          # max number of GPUs available
_MAX_GPU_LOAD = 0.05        # max load to be considered as available
_MAX_GPU_MEM_USED = 0.2     # max memory usage to be considered as available


_logger = logging.getLogger(__name__)


def get_computation_devices(
        preferred_gpu_list: Optional[List[int]],
        multi_gpu_flag: bool,
) -> List[torch.device]:

    # use CPU when GPUs are not preferred or not available
    if (preferred_gpu_list is None) \
            or (len(preferred_gpu_list) == 0) \
            or (not torch.cuda.is_available()):
        return [torch.device('cpu'), ]

    # GPUs are preferred and available
    else:
        # get all available GPU indexes
        _available_gpu_list: List[int]
        try:
            # by default, use GPU utility package with load and memory usage
            # specification so that the 'available' GPUs are actually ready
            # for deep learning runs (https://github.com/anderskm/gputil)
            from GPUtil import getAvailable
            _available_gpu_list = getAvailable(
                limit=_MAX_NUM_GPUS,
                maxLoad=_MAX_GPU_LOAD,
                maxMemory=_MAX_GPU_MEM_USED,
            )
        except (ImportError, ModuleNotFoundError):
            # assume all GPUs are good to use without GPUtil package
            _available_gpu_list = list(range(torch.cuda.device_count()))
            _logger.warning(
                f'GPUtil (https://github.com/anderskm/gputil) not installed.'
                f'Assuming all GPUs ({_available_gpu_list}) are available '
                f'and ready for training ... '
            )

        # get the overlap between the preferred and the available GPUs
        _gpus = \
            [_g for _g in _available_gpu_list if _g in preferred_gpu_list]

        # use CPU if there is no preferred GPUs that are available
        if len(_gpus) == 0:
            return [torch.device('cpu'), ]
        # otherwise return one or all GPUs depending on the multi-GPU flag
        else:
            return [torch.device(f'cuda:{_g}') for _g in _gpus] \
                if multi_gpu_flag else [torch.device(f'cuda:{_gpus[0]}'), ]
