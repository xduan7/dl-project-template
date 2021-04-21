import inspect

from types import ModuleType
from typing import Any, Callable, Dict, Optional, Type

from src.utilities import get_closest_match


def get_object_from_module(
        object_name: str,
        module: ModuleType,
        predicate: Optional[Callable] = None,
) -> Any:
    """Get an object by name from a module with given predicate.

    This function will inspect all the members in a module no matter
    the type, and then try and search for a member that is (1)
    equal or similar to the desired object name, and (2) passes the
    predicate, which means that ``predicate(object)`` must be `True`.
    The function raises `ValueError` if no such object found.

    Args:
        object_name: A string as the name of target object.
        module: The module to inspect and search from.
        predicate: An optional predicate function for the searching
            criteria and therefore restricts returned object.
            Non-applicable if sets to `None`.

    Returns:
        A object with a name equal or similar to the target name,
        that passes the predicate if given.

    Raises:
        ValueError: No match found in the module.

    """
    _object_dict: Dict[str, Any] = {
        _module_member[0]: _module_member[1]
        for _module_member in inspect.getmembers(module, predicate)
    }
    _object_names_in_module = list(_object_dict.keys())
    if object_name in _object_names_in_module:
        return _object_dict[object_name]
    try:
        _closest_match_object_name = \
            get_closest_match(object_name, _object_names_in_module)
        return _object_dict[_closest_match_object_name]
    except ValueError:
        _module_name: str = module.__name__
        _error_msg = \
            f'Module \'{_module_name}\' has no object with ' \
            f'a name equal or similar to \'{object_name}\'.'
        raise ValueError(_error_msg)


def get_class_from_module(
        class_name: str,
        module: ModuleType,
) -> Type:
    """Get an class by name from a module.

    Args:
        class_name: A string as the name of target class.
        module: The module to inspect and search from.

    Returns:
        A class with a name equal or similar to the target name.

    Raises:
        ValueError: No match found in the module.

    """
    _class: Type = get_object_from_module(
        object_name=class_name,
        module=module,
        predicate=inspect.isclass,
    )
    return _class


def get_function_from_module(
        func_name: str,
        module: ModuleType,
) -> Callable:
    """Get an function by name from a module.

    Args:
        func_name: A string as the name of target function.
        module: The module to inspect and search from.

    Returns:
        A function with a name equal or similar to the target name.

    Raises:
        ValueError: No match found in the module.

    """
    _func: Callable = get_object_from_module(
        object_name=func_name,
        module=module,
        predicate=inspect.isfunction,
    )
    return _func
