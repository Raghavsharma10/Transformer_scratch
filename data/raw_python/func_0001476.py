def _representable(value: Any) -> bool:
    """
    Check whether we want to represent the value in the error message on contract breach.

    We do not want to represent classes, methods, modules and functions.

    :param value: value related to an AST node
    :return: True if we want to represent it in the violation error
    """
    return not inspect.isclass(value) and not inspect.isfunction(value) and not inspect.ismethod(value) and not \
        inspect.ismodule(value) and not inspect.isbuiltin(value)