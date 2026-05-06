def cyber_observable_check(original_function):
    """Decorator for functions that require cyber observable data.
    """
    def new_function(*args, **kwargs):
        if not has_cyber_observable_data(args[0]):
            return
        func = original_function(*args, **kwargs)
        if isinstance(func, Iterable):
            for x in original_function(*args, **kwargs):
                yield x
    new_function.__name__ = original_function.__name__
    return new_function