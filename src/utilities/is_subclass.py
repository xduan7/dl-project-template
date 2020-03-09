"""
File Name:          is_subclass.py
Project:            dl-project-template

File Description:

"""
from inspect import isclass
from typing import Any, Type


def is_subclass(
        subclass: Any,
        base_class: Type,
) -> bool:
    return isclass(subclass) \
           and issubclass(subclass, base_class) \
           and (subclass != base_class)
