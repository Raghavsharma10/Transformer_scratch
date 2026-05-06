def cached(fn, size=32):
    ''' this decorator creates a type safe lru_cache
    around the decorated function. Unlike
    functools.lru_cache, this will not crash when
    unhashable arguments are passed to the function'''
    assert callable(fn)
    assert isinstance(size, int)
    return overload(fn)(lru_cache(size, typed=True)(fn))