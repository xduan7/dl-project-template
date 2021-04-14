import os
import warnings
from typing import List, Union

import torch
from GPUtil import getAvailable


Device = Union[int, torch.device]


# Criteria for GPU availability
_MAX_NUM_GPUS = float('inf')    # max number of GPUs available
_MAX_GPU_LOAD = 0.2             # max load to be considered as available
_MAX_GPU_MEM_USED = 0.2         # max memory usage considered as available


def __get_available_gpus() -> List[Device]:
    """Get the indices of all empty and available GPUs for PyTorch.

    The function will first fetch the indices of all empty GPUs with
    GPUtil by checking GPU load and memory usage. Then it modifies the
    indices to correspond to `CUDA_VISIBLE_DEVICES`. Finally,
    it checks with PyTorch with ``torch.cuda.get_device_properties``
    to make sure that PyTorch has access to all the GPUs whose indices
    are returned.

    Please refer to ``__reindex_available_gpus_by_visible_devices``
    for further details on reindexing with `CUDA_VISIBLE_DEVICES`.

    Returns: A list of GPU indices or devices that PyTorch has access to.

    """
    _available_gpus = getAvailable(
        limit=_MAX_NUM_GPUS,
        maxLoad=_MAX_GPU_LOAD,
        maxMemory=_MAX_GPU_MEM_USED,
    )
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        _available_gpus = \
            __reindex_available_gpus_by_visible_devices(_available_gpus)
    return __restrict_available_gpus_by_torch(_available_gpus)


def __reindex_available_gpus_by_visible_devices(
        available_gpus: List[Device],
) -> List[Device]:
    """Re-index the list of empty GPUs with `CUDA_VISIBLE_DEVICES`.

    The empty GPUs are returned from ``getAvailable``, which uses
    Nvidia APIs directly and ignores current `CUDA_VISIBLE_DEVICES`.
    However, PyTorch can only have access to visible devices, and the
    indices of which are different from the global ones. This
    function helps re-indexing the 'global' indices to the ones that
    are applicable to PyTorch.

    Example:
        >>> # CUDA_VISIBLE_DEVICES = [2, 3]
        >>> getAvailable(...)
        [0, 1, 2]
        >>> # global indices of available GPUs is [2]
        >>> # PyTorch only has access to 2 GPUs, indexed with [0, 1]
        >>> # indices of available GPUs after re-indexing is [0]
        >>> __reindex_available_gpus_by_visible_devices([0, 1, 2])
        [0]

    Args:
        available_gpus: A list of global (of all GPUs and ignores
        `CUDA_VISIBLE_DEVICES`) indices empty GPUs.

    Returns: A list of indices empty GPUs applicable to PyTorch.

    """
    _cuda_visible_devices = [
        int(__d) for __d in
        os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    ]
    _reindexed_available_gpus = []
    for __available_gpu in available_gpus:
        if __available_gpu in _cuda_visible_devices:
            _reindexed_available_gpus.append(
                _cuda_visible_devices.index(__available_gpu))
    return _reindexed_available_gpus


def __restrict_available_gpus_by_torch(
        available_gpus: List[Device],
) -> List[Device]:
    """Restrict the indices of empty and visible GPUs by checking with
    PyTorch API for accessibility.

    """
    _restricted_available_gpus: List[Device] = []
    for __available_gpu in available_gpus:
        try:
            torch.cuda.get_device_properties(__available_gpu)
        except AssertionError:
            _warning_msg = \
                f'CUDA device {__available_gpu}, despite being ' \
                f'available by GPUtil and `CUDA_VISIBLE_DEVICES`, ' \
                f'is not accessible by PyTorch.'
            warnings.warn(_warning_msg)
            print(_warning_msg)
        else:
            _restricted_available_gpus.append(__available_gpu)
    return _restricted_available_gpus


def get_computation_devices() -> List[torch.device]:
    """Get the available computation devices (CPU or GPUs) for PyTorch.

    This function first get the global indices for all the empty GPUs,
    with load less than ``_MAX_GPU_LOAD`` and memory usage less than
    ``_MAX_GPU_MEM_USED``. Global indices are essentially the ones
    from `nvidia-smi`. Then the indices of the empty GPUs will be
    modified to make sure they are accessible by PyTorch. The
    function will return all available GPUs in a list of
    ``torch.device`` unless there is none, in which case the function
    returns ``[torch.device('cpu')]``.

    Returns: A list of computation devices (``torch.device``),
        made of either  one or more GPUs or a single CPU device.

    """
    if not torch.cuda.is_available():
        return [torch.device('cpu')]
    _available_gpus = __get_available_gpus()
    return [torch.device('cpu'), ] if len(_available_gpus) == 0 \
        else [torch.device(f'cuda:{_g}') for _g in _available_gpus]
