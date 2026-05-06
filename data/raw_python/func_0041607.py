def timeit(func):
    """
    Simple decorator to time functions

    :param func: function to decorate
    :type func: callable
    :return: wrapped function
    :rtype: callable
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.time()

        result = func(*args, **kwargs)

        elapsed = time.time() - start
        LOGGER.info('%s took %s seconds to complete', func.__name__, round(elapsed, 2))
        return result

    return _wrapper