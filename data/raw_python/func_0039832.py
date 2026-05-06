def cached(func):
    """
    A decorator function to cache values. It uses the decorated
    function's arguments as the keys to determine if the function
    has been called previously.
    """
    cache = {}

    @f.wraps(func)
    def wrapper(*args, **kwargs):
        key = func.__name__ + str(sorted(args)) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper