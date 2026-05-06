def memoize(fun):
    """Memoizes return values of the decorated function.

    Similar to l0cache, but the cache persists for the duration of the process, unless clear_cache()
    is called on the function.

    """
    argspec = inspect2.getfullargspec(fun)
    arg_names = argspec.args + argspec.kwonlyargs
    kwargs_defaults = get_kwargs_defaults(argspec)

    def cache_key(args, kwargs):
        return get_args_tuple(args, kwargs, arg_names, kwargs_defaults)

    @functools.wraps(fun)
    def new_fun(*args, **kwargs):
        k = cache_key(args, kwargs)
        if k not in new_fun.__cache:
            new_fun.__cache[k] = fun(*args, **kwargs)
        return new_fun.__cache[k]

    def clear_cache():
        """Removes all cached values for this function."""
        new_fun.__cache.clear()

    new_fun.__cache = {}
    new_fun.clear_cache = clear_cache
    return new_fun