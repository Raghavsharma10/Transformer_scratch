def memoize_with_ttl(ttl_secs=60 * 60 * 24):
    """Memoizes return values of the decorated function for a given time-to-live.

    Similar to l0cache, but the cache persists for the duration of the process, unless clear_cache()
    is called on the function or the time-to-live expires. By default, the time-to-live is set to
    24 hours.

    """

    error_msg = (
        "Incorrect usage of qcore.caching.memoize_with_ttl: "
        "ttl_secs must be a positive integer."
    )
    assert_is_instance(ttl_secs, six.integer_types, error_msg)
    assert_gt(ttl_secs, 0, error_msg)

    def cache_fun(fun):
        argspec = inspect2.getfullargspec(fun)
        arg_names = argspec.args + argspec.kwonlyargs
        kwargs_defaults = get_kwargs_defaults(argspec)

        def cache_key(args, kwargs):
            return repr(get_args_tuple(args, kwargs, arg_names, kwargs_defaults))

        @functools.wraps(fun)
        def new_fun(*args, **kwargs):
            k = cache_key(args, kwargs)
            current_time = int(time.time())

            # k is not in the cache; perform the function and cache the result.
            if k not in new_fun.__cache or k not in new_fun.__cache_times:
                new_fun.__cache[k] = fun(*args, **kwargs)
                new_fun.__cache_times[k] = current_time
                return new_fun.__cache[k]

            # k is in the cache at this point. Check if the ttl has expired;
            # if so, recompute the value and cache it.
            cache_time = new_fun.__cache_times[k]
            if current_time - cache_time > ttl_secs:
                new_fun.__cache[k] = fun(*args, **kwargs)
                new_fun.__cache_times[k] = current_time

            # finally, return the cached result.
            return new_fun.__cache[k]

        def clear_cache():
            """Removes all cached values for this function."""
            new_fun.__cache.clear()
            new_fun.__cache_times.clear()

        def dirty(*args, **kwargs):
            """Dirties the function for a given set of arguments."""
            k = cache_key(args, kwargs)
            new_fun.__cache.pop(k, None)
            new_fun.__cache_times.pop(k, None)

        new_fun.__cache = {}
        new_fun.__cache_times = {}
        new_fun.clear_cache = clear_cache
        new_fun.dirty = dirty
        return new_fun

    return cache_fun