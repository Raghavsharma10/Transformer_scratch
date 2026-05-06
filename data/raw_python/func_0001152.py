def with_log(func):
    """Automatically adds a named logger to a function upon function call.

    :param func: Function to decorate.

    :return: Decorated function.
    :rtype: function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Inject `log` argument into wrapped function.

        :param list args: Pass through all positional arguments.
        :param dict kwargs: Pass through all keyword arguments.
        """
        decorator_logger = logging.getLogger('@with_log')
        decorator_logger.debug('Entering %s() function call.', func.__name__)
        log = kwargs.get('log', logging.getLogger(func.__name__))
        try:
            ret = func(log=log, *args, **kwargs)
        finally:
            decorator_logger.debug('Leaving %s() function call.', func.__name__)
        return ret
    return wrapper