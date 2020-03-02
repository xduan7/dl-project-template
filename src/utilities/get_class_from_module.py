"""
File Name:          get_class_from_module.py
Project:            dl-project-template

File Description:

"""
import logging
import inspect
from difflib import get_close_matches
from types import ModuleType
from typing import Dict


_LOGGER = logging.getLogger(__name__)


def get_class_from_module(
        class_name: str,
        module: ModuleType,
) -> type:
    """get a class as type instance with given name from the given module

    This function will first inspect the module and get all the classes (
    their names and types), then if the given class name matches any of the
    class name inside the module, the function will simply return the class.
    Otherwise, the functions will check the close matches. If there is one
    close match inside the whole module, the function will take it and
    proceed with warning. If there are more than one close matches,
    the function will raise value error indicating that the given name
    imposes ambiguity.

    :param class_name: name of the target class in string
    :param module: module to inspect and search from
    :return: class (as python type instance) if the given class name (1)
    matches one of the classes inside the module exactly; or (2) has a
    single close match with all the classes inside the module
    """

    _class_dict: Dict[str, type] = {
        _module_member[0]: _module_member[1]
        for _module_member in inspect.getmembers(module, inspect.isclass)
    }

    _class: type
    if class_name in _class_dict.keys():
        _class = _class_dict[class_name]
    else:
        _close_match_names = \
            get_close_matches(class_name, _class_dict.keys())
        if len(_close_match_names) == 1:
            _warning_msg = \
                f'Do you mean to use class \'{_close_match_names[0]}\' ' \
                f'instead of \'{class_name}\' from module \'{module}\'? ' \
                f'Continuing with \'{_close_match_names[0]}\' class ...'
            _LOGGER.warning(_warning_msg)
            _class = _class_dict[_close_match_names[0]]
        elif len(_close_match_names) > 1:
            _error_msg = \
                f'Ambiguity with class name \'{class_name}\' . Please pick ' \
                f'from any of these classes from module \'{module}\': ' \
                f'{_close_match_names}'
            raise ValueError(_error_msg)
        else:
            _error_msg = \
                f'Module \'{module}\' has no class name ' \
                f'similar to \'{class_name}\'.'
            raise AttributeError(_error_msg)

    return _class
