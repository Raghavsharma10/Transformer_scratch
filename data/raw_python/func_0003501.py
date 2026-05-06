def callwith(context_manager):
    """
    A decorator to wrap execution of function with a context manager.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            with context_manager:
                return func(*args, **kwds)
        return wrapper
    return decorator