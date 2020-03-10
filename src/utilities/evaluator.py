"""
File Name:          evaluator.py
Project:            dl-project-template

File Description:

"""
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union
from functools import partial

import torch
import sklearn
import numpy as np
import pandas as pd

from src.utilities import get_function_from_module, get_valid_kwargs


_LOGGER = logging.getLogger(__name__)


class Evaluator:

    def __init__(
            self,
            metrics: Optional[List[str]] = None,
    ):
        self._metrics: Dict[str, Callable] = {}
        self._data: Dict[int, Dict[str, List[float]]] = {}
        self._results: Dict[int, Dict[str, Any]] = {}

        for _metric in metrics:
            self.add_metric(_metric)

    def __str__(self) -> str:
        return str(self.to_dataframe())

    def __index__(
            self,
            epoch: int,
    ) -> Optional[Dict[str, List[float]]]:
        if epoch in self._data:
            return self._data[epoch]
        else:
            _warning_msg = \
                f'Epoch {epoch} is not logged into evaluator.'
            return None

    def add_metric(
            self,
            metric: Union[str, Callable],
            metric_kwargs: Optional[Dict[str, Any]] = None,
    ):
        _func: Callable = \
            get_function_from_module(metric, sklearn.metrics) \
            if isinstance(metric, str) else metric
        _name = _func.__name__

        if metric_kwargs:
            # note that partial will not check if the given parameters
            # exist in the function signature; need to check manually
            _given_arg_keys: Set[str] = set(metric_kwargs.keys())
            _complete_arg_keys: Set[str] = \
                set(inspect.getfullargspec(_func).args)
            try:
                assert _complete_arg_keys >= _given_arg_keys
            except AssertionError:
                _invalid_arg_keys = _given_arg_keys - _complete_arg_keys
                _warning_msg = \
                    f'Given keyword-arguments for metric function ' \
                    f'{_func.__name__} contains the following invalid ' \
                    f'keys {_invalid_arg_keys}. ' \
                    f'Continuing by ignoring them ... '
                _LOGGER.warning(_warning_msg)

                metric_kwargs = {
                    _k: _v for _k, _v in metric_kwargs.items()
                    if _k in _complete_arg_keys
                }
            _func = partial(_func, **metric_kwargs)
            _kwargs_str = \
                ''.join(f'\'{_k}\'={_v}' for _k, _v in metric_kwargs.items())
            _name = _name + f'({_kwargs_str})'

        if _name in self._metrics:
            _warning_msg = \
                f'Metric function with the name {_name} already exists in ' \
                f'the evaluator, and therefore will be ignored. ' \
                f'Please check if this is a duplicate.'
            _LOGGER.warning(_warning_msg)
        else:
            self._metrics[_name] = _func

    def add_data(
            self,
            epoch: int,
            data: Dict[str, torch.Tensor],
    ):
        if epoch not in self._data:
            self._data[epoch] = {}

        for _data_key, _pt_tensor in data.items():
            _np_array: np.ndarray = _pt_tensor.detach().cpu().numpy()
            # use list of np.arrays for faster extend
            _data_list = list(_np_array)
            if _data_key in self._data[epoch]:
                self._data[epoch][_data_key].extend(_data_list)
            else:
                self._data[epoch][_data_key] = _data_list

    def _calculate_metric(
            self,
            epoch: int,
            metric_name: str,
    ):
        if epoch not in self._results:
            self._results[epoch] = {}
        elif metric_name in self._results[epoch]:
            return

        _metric_func: Callable = self._metrics[metric_name]
        _valid_kwargs = \
            get_valid_kwargs(func=_metric_func, kwargs=self._data[epoch])

        try:
            self._results[epoch][metric_name] = _metric_func(**_valid_kwargs)
        except Exception as _exception:
            _error_msg = \
                f'Encountered error when calculating metric {metric_name} ' \
                f'for epoch {epoch}. Exception message\n: {str(_exception)}'
            _LOGGER.error(_error_msg)
            self._results[epoch][metric_name] = np.NaN

    def calculate_metrics(self):
        _epochs: List[int] = list(self._data.keys())
        _metric_names: List[str] = list(self._metrics.keys())
        for _epoch in _epochs:
            for _metric_name in _metric_names:
                self._calculate_metric(_epoch, _metric_name)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._results, orient='index')
