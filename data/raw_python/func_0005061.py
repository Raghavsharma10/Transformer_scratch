def tool(name):
    # type: (str) -> FunctionType
    """ Decorator for defining lint tools.

    Args:
        name (str):
            The name of the tool. This name will be used to identify the tool
            in `pelconf.yaml`.
    """
    global g_tools

    def decorator(fn):  # pylint: disable=missing-docstring
        # type: (FunctionType) -> FunctionType
        g_tools[name] = fn
        return fn

    return decorator