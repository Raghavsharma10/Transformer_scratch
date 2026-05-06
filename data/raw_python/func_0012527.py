def catch_it(exceptions=(Exception, ), default=None, handler=None):
    """
    If the function encounters an exception, catch it, and
    return the specified default or sent to a handler function instead.

    .. code :: python

        def handle_error(exception, func, *args, **kwargs):
            print(f"{func.__name__} raised {exception} when called with {args}")

        @reusables.catch_it(handler=err_func)
        def will_raise(message="Hello")
            raise Exception(message)


    :param exceptions: tuple of exceptions to catch
    :param default: what to return if the exception is caught
    :param handler: function to send exception, func, *args and **kwargs
    """
    def func_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as err:
                if handler:
                    return handler(err, func, *args, **kwargs)
                return default
        return wrapper
    return func_wrapper