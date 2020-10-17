"""
File Name:          get_module_summary.py
Project:            dl-project-template

File Description:

    Keras style model summary for PyTorch
    reference: https://github.com/sksq96/pytorch-summary

    This implementation differs from the original one at the reference
    repo in the following ways:
    - the function only accept input, either a tensor or a sequence of them;
      this alleviates the trouble of specifying dtype, batch size,
      and slicing inside a single batch
    - fixed the input size calculation
    - fixed the buffer size calculation
    - optimization for coding style and computation ...

"""
import logging
from collections import OrderedDict
from typing import List, Sequence, Tuple, Union

import torch
import numpy as np
import torch.nn as nn

from torch.utils.hooks import RemovableHandle


_LOGGER = logging.getLogger(__name__)
_LAYER_NAME_LEN: int = 32
_OUTPUT_SHAPE_LEN: int = 48
_NUM_PARAM_LEN: int = 20
_HEADER_LEN: int = _LAYER_NAME_LEN + _OUTPUT_SHAPE_LEN + _NUM_PARAM_LEN


def get_module_summary(
        module: nn.Module,
        batch_input: Union[torch.Tensor, Sequence[torch.Tensor]],
) -> Tuple[OrderedDict, str]:
    """get the PyTorch module summary with given batched input

    :param module: module to be summarized
    :type module: nn.Module
    :param batch_input: batched input
    :type batch_input: torch.Tensor or a sequence of torch.Tensor
    :return: an ordered dict for module in the ordering of forward,
    and a string summarized info
    :rtype: Tuple[OrderedDict, str]
    """
    _summary = OrderedDict()
    _summary_hooks: List[RemovableHandle] = []

    # function for data collection during a forward
    def _get_shape(__data):
        try:
            return list(__data.size())
        except AttributeError:
            return [_get_shape(__d) for __d in __data]

    def _summary_hook(__module, __input, __output):
        __module_class_name: str = \
            str(__module.__class__).split('.')[-1].split('\'')[0]
        __module_index: int = len(_summary)
        __module_key_: str = f'{__module_class_name}-{__module_index}'

        _summary[__module_key_] = OrderedDict()
        _summary[__module_key_]['input_shape'] = _get_shape(__input)
        _summary[__module_key_]['output_shape'] = _get_shape(__output)

        _num_params: int = 0
        if hasattr(__module, 'weight') \
                and hasattr(__module.weight, 'size'):
            _num_params += np.prod(list(__module.weight.size()))
            _summary[__module_key_]['trainable'] = \
                __module.weight.requires_grad
        if hasattr(__module, 'bias') \
                and hasattr(__module.bias, 'size'):
            _num_params += np.prod(list(__module.bias.size()))
        if hasattr(__module, '_buffers'):
            _num_params += np.sum([
               np.prod(__b.size())
                for __b in list(__module._buffers.values())
            ])
        _summary[__module_key_]['num_params'] = int(_num_params)

    def _register_summary_hook(__module):
        if ((not isinstance(__module, nn.Sequential))
                and (not isinstance(__module, nn.ModuleList))):
            _summary_hooks.append(
                __module.register_forward_hook(_summary_hook))

    # register hook, apply a forward run, and remove all the hooks
    module.apply(_register_summary_hook)
    try:
        module(batch_input)
    except TypeError:
        _warning_msg = \
            f'Failed to run model forward with batch input directly; ' \
            f're-trying with expanded batch input ... '
        _LOGGER.warning(_warning_msg)
        module(*batch_input)
    for __h in _summary_hooks:
        __h.remove()

    _header = \
        f'{"Layer (type)":>{_LAYER_NAME_LEN}}' \
        f'{"Output Shape":>{_OUTPUT_SHAPE_LEN}}' \
        f'{"Param #":>{_NUM_PARAM_LEN}}\n'
    _summary_str = \
        ('-' * _HEADER_LEN + '\n' + _header + '=' * _HEADER_LEN + '\n')

    def _get_size(__shape):
        if isinstance(__shape[0], int):
            return np.prod(__shape)
        else:
            return np.sum([_get_size(__s) for __s in __shape])

    _total_num_params, _num_trainable_params, _num_layer_outputs = 0, 0, 0
    for __module_key in _summary:
        __module__output_shape = str(_summary[__module_key]["output_shape"])
        __module_summary_str = \
            f'{__module_key:>{_LAYER_NAME_LEN}}' \
            f'{__module__output_shape:>{_OUTPUT_SHAPE_LEN}}' \
            f'{_summary[__module_key]["num_params"]:>{_NUM_PARAM_LEN},}\n'
        _summary_str += __module_summary_str

        _total_num_params += _summary[__module_key]["num_params"]
        _num_layer_outputs += _get_size(_summary[__module_key]['output_shape'])
        if (('trainable' in _summary[__module_key]) and
                _summary[__module_key]["trainable"]):
            _num_trainable_params += _summary[__module_key]['num_params']

    # assume 4 bytes per number (float on cuda) and size in mb
    __float_in_mb = 4. / (1024 ** 2.)
    if isinstance(batch_input, torch.Tensor):
        _input_size_in_mb = np.prod(batch_input.size()) * __float_in_mb
    else:
        _input_size_in_mb = np.sum(
            [np.prod(__i.size()) for __i in batch_input]) * __float_in_mb

    # forward and backward, hence the 2*
    _layer_output_size_in_mb = 2. * _num_layer_outputs * __float_in_mb
    _total_params_size_in_mb = _total_num_params * __float_in_mb
    _total_size_in_mb = \
        _input_size_in_mb + \
        _layer_output_size_in_mb + \
        _total_params_size_in_mb

    _summary_str += '=' * _HEADER_LEN + '\n'
    _summary_str += f'Total params:{_total_num_params:,}\n'
    _summary_str += f'Trainable params:{_num_trainable_params:,}\n'
    _summary_str += f'Non-trainable params:' \
                    f'{_total_num_params - _num_trainable_params:,}\n'
    _summary_str += '-' * _HEADER_LEN + '\n'
    _summary_str += f'Input size (MB):{_input_size_in_mb:0.2f}\n'
    _summary_str += f'Forward/backward pass size (MB): ' \
                    f'{_layer_output_size_in_mb:0.2f}\n'
    _summary_str += f'Params size (MB):{_total_params_size_in_mb:0.2f}\n'
    _summary_str += f'Estimated Total Size (MB):{_total_size_in_mb:0.2f}\n'
    _summary_str += '-' * _HEADER_LEN + '\n'

    return _summary, _summary_str
