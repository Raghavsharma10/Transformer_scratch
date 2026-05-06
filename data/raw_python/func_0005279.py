def typed_lru(maxsize, types=None):
    """ :func:functools.lru_cache wrapper which allows you to prevent object
        types outside of @types from being cached.

        The main use case for this is preventing unhashable type errors when
        you still want to cache some results.
        ..
            from vital.cache import typed_lru

            @typed_lru(300, (str, int))
            def some_expensive_func():
                pass

            @typed_lru(300, str)
            def some_expensive_func2():
                pass

            @typed_lru(300, collections.Hashable)
            def some_expensive_func3():
                pass
        ..
    """
    types = types or collections.Hashable

    def lru(obj):
        @lru_cache(maxsize)
        def _lru_cache(*args, **kwargs):
            return obj(*args, **kwargs)

        @wraps(obj)
        def _convenience(*args, **kwargs):
            broken = False
            for arg in args:
                if not isinstance(arg, types):
                    broken = True
                    break
            for arg, val in kwargs.items():
                if not isinstance(arg, types) and isinstance(val, types):
                    broken = True
                    break
            if not broken:
                try:
                    return _lru_cache(*args, **kwargs)
                except TypeError:
                    return obj(*args, **kwargs)
            return obj(*args, **kwargs)
        return _convenience
    return lru