def lru_cache(maxsize=128, key_fn=None):
    """Decorator that adds an LRU cache of size maxsize to the decorated function.

    maxsize is the number of different keys cache can accomodate.
    key_fn is the function that builds key from args. The default key function
    creates a tuple out of args and kwargs. If you use the default, there is no reason
    not to use functools.lru_cache directly.

    Possible use cases:
    - Your cache key is very large, so you don't want to keep the whole key in memory.
    - The function takes some arguments that don't affect the result.

    """

    def decorator(fn):
        cache = LRUCache(maxsize)
        argspec = inspect2.getfullargspec(fn)
        arg_names = argspec.args[1:] + argspec.kwonlyargs  # remove self
        kwargs_defaults = get_kwargs_defaults(argspec)

        cache_key = key_fn
        if cache_key is None:

            def cache_key(args, kwargs):
                return get_args_tuple(args, kwargs, arg_names, kwargs_defaults)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = cache_key(args, kwargs)
            try:
                return cache[key]
            except KeyError:
                value = fn(*args, **kwargs)
                cache[key] = value
                return value

        wrapper.clear = cache.clear

        return wrapper

    return decorator