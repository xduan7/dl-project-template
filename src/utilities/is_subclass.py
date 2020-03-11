"""
File Name:          is_subclass.py
Project:            dl-project-template

File Description:

    This file implement a simple replacement for subclass checking

"""
from inspect import isclass
from typing import Any, Type


def is_subclass(
        subclass: Any,
        base_class: Type,
) -> bool:
    """check if a class a subclass of another class

    Improve the generic 'issubclass' function by adding: (1) check if the
    subclass is actually a class, and (2) make sure the subclass is not the
    same as base class.

    :param subclass: subclass candidate for checking
    :param base_class: base/parent class
    :return: bool indicator for subclass checking
    """
    return isclass(subclass) \
        and issubclass(subclass, base_class) \
        and (subclass != base_class)
