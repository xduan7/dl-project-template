from inspect import isclass
from typing import Any, Type


def is_subclass(
        class_: Any,
        base_class: Type,
) -> bool:
    """Check if a class strictly a subclass of the base class.

    Args:
        class_: A class of any type.
        base_class: A potential base class for the given class.

    Returns:
        Boolean indicating if the class is the subclass of the base class.

    """
    return isclass(class_) \
        and issubclass(class_, base_class) \
        and (class_ != base_class)
