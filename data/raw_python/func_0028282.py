def config_ctx(func):
    """
    Decorator that makes decorated function use ConfigurationContext instead \
    of Context instance.

    :param func: Decorated function.

    :return: Decorated function.
    """
    # Create ConfigurationContext subclass
    class _ConfigurationContext(ConfigurationContext):
        # Set command name for the context class
        cmd = func.__name__

        # Set function name for the context class
        fun = func.__name__

    # Store the created context class with the decorated function
    func._context_class = _ConfigurationContext  # pylint: disable=W0212

    # Return the decorated function
    return func