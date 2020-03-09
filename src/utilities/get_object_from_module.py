"""
File Name:          get_object_from_module.py
Project:            dl-project-template

File Description:

"""
import logging
import inspect

from typing import Dict, Optional, Callable, Any, Type
from types import ModuleType
from difflib import get_close_matches


_LOGGER = logging.getLogger(__name__)


def get_object_from_module(
        object_name: str,
        module: ModuleType,
        predicate: Optional[Callable] = None,
) -> Any:
    """get an object from a module with given name and predicate

    This function will inspect the module and get all the objects (that
    passes the predicate if given) and their names, and check the given
    object name against them. The function will return the object if either
    there is an object with the exact name, or there is a single close
    match. Otherwise, the function will raise ValueError.

    :param object_name: name of the target object
    :param module: module to inspect and search from
    :param predicate: optional predicate function to limit the type (e.g.
    function, class, etc.) of the target object. Defaults to None
    :return: matched object if the given name matches (exactly or closely
    without ambiguity) any objects in the module
    """

    _object_dict: Dict[str, Any] = {
        _module_member[0]: _module_member[1]
        for _module_member in inspect.getmembers(module, predicate)
    } if predicate else {
        _module_member[0]: _module_member[1]
        for _module_member in inspect.getmembers(module)
    }

    if object_name in _object_dict.keys():
        _object = _object_dict[object_name]
    else:
        _close_match_names = \
            get_close_matches(object_name, _object_dict.keys())
        _module_name: str = module.__name__

        if len(_close_match_names) == 1:
            _close_match_name = str(_close_match_names[0])
            _warning_msg = \
                f'Do you mean to use object \'{_close_match_name}\' ' \
                f'instead of \'{object_name}\' from module ' \
                f'\'{_module_name}\'? Continuing with object ' \
                f'\'{_close_match_name}\' ...'
            _LOGGER.warning(_warning_msg)
            _object = _object_dict[_close_match_name]
        elif len(_close_match_names) > 1:
            _error_msg = \
                f'Ambiguity with object name \'{object_name}\' . ' \
                f'Please pick from any of these objects from module ' \
                f'\'{_module_name}\': {_close_match_names}'
            raise ValueError(_error_msg)
        else:
            _error_msg = \
                f'Module \'{_module_name}\' has no object with name ' \
                f'similar to \'{object_name}\'.'
            raise ValueError(_error_msg)

    return _object


def get_class_from_module(
        class_name: str,
        module: ModuleType,
) -> Type:
    """get an class from a module with given name

    Please refer to the function 'src.utilities.get_object_from_module' for
    detailed execution and behaviour.

    :param class_name: name of the target class
    :param module: module to inspect and search from
    :return: matched class if the given name matches (exactly or closely
    without ambiguity) any classes in the module
    """

    _class: Type = get_object_from_module(
        object_name=class_name,
        module=module,
        predicate=inspect.isclass,
    )

    return _class


def get_function_from_module(
        function_name: str,
        module: ModuleType,
) -> Callable:
    """get an function from a module with given name

    Please refer to the function 'src.utilities.get_object_from_module' for
    detailed execution and behaviour.

    :param function_name: name of the target function
    :param module: module to inspect and search from
    :return: matched function if the given name matches (exactly or closely
    without ambiguity) any functions in the module
    """

    _function: Callable = get_object_from_module(
        object_name=function_name,
        module=module,
        predicate=inspect.isfunction,
    )

    return _function
