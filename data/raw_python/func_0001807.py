def plugin_method(*plugin_names):
    """Plugin Method decorator.
    Signs a web handler function with the plugins to be applied as attributes.

    Args:
        plugin_names (list): A list of plugin callable names

    Returns:
        A wrapped handler callable.

    Examples:
        >>> @plugin_method('json', 'bill')
        ... def method():
        ...     return "Hello!"
        ...
        >>> print method.json
        True
        >>> print method.bill
        True

    """
    def wrapper(callable_obj):
        for plugin_name in plugin_names:
            if not hasattr(callable_obj, plugin_name):
                setattr(callable_obj, plugin_name, True)
        return callable_obj
    return wrapper