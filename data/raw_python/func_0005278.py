def local_lru(obj):
    """ Property that maps to a key in a local dict-like attribute.
        self._cache must be an OrderedDict
        self._cache_size must be defined as LRU size
        ..
        class Foo(object):

            def __init__(self, cache_size=5000):
                self._cache = OrderedDict()
                self._cache_size = cache_size

            @local_lru
            def expensive_meth(self, arg):
                pass
        ..
    """
    @wraps(obj)
    def memoizer(*args, **kwargs):
        instance = args[0]
        lru_size = instance._cache_size
        if lru_size:
            cache = instance._cache
            key = str((args, kwargs))
            try:
                r = cache.pop(key)
                cache[key] = r
            except KeyError:
                if len(cache) >= lru_size:
                    cache.popitem(last=False)
                r = cache[key] = obj(*args, **kwargs)
            return r
        return obj(*args, **kwargs)
    return memoizer