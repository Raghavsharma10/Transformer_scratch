def unique(max_retries=10, wait=0, alt_return="-no_alt_return-",
           exception=Exception, error_text=None):
    """
    Wrapper. Makes sure the function's return value has not been returned before
    or else it run with the same inputs again.

    .. code: python

        import reusables
        import random

        @reusables.unique(max_retries=100)
        def poor_uuid():
            return random.randint(0, 10)

        print([poor_uuid() for _ in range(10)])
        # [8, 9, 6, 3, 0, 7, 2, 5, 4, 10]

        print([poor_uuid() for _ in range(100)])
        # Exception: No result was unique

    Message format options: {func} {args} {kwargs}

    :param max_retries: int of number of retries to attempt before failing
    :param wait: float of seconds to wait between each try, defaults to 0
    :param exception: Exception type of raise
    :param error_text: text of the exception
    :param alt_return: if specified, an exception is not raised on failure,
     instead the provided value of any type of will be returned
    """
    def func_wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = (error_text if error_text else
                   "No result was unique for function '{func}'")
            if not error_text:
                msg = _add_args(msg, *args, **kwargs)
            for i in range(max_retries):
                value = func(*args, **kwargs)
                if value not in unique_cache[func.__name__]:
                    unique_cache[func.__name__].append(value)
                    return value
                if wait:
                    time.sleep(wait)
            else:
                if alt_return != "-no_alt_return-":
                    return alt_return
                raise exception(msg.format(func=func.__name__,
                                           args=args, kwargs=kwargs))
        return wrapper
    return func_wrap