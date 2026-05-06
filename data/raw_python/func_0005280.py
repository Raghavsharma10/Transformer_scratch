def local_expiring_lru(obj):
    """ Property that maps to a key in a local dict-like attribute.
        self._cache must be an OrderedDict
        self._cache_size must be defined as LRU size
        self._cache_ttl is the expiration time in seconds
        ..
        class Foo(object):

            def __init__(self, cache_size=5000, cache_ttl=600):
                self._cache = OrderedDict()
                self._cache_size = cache_size
                self._cache_ttl = cache_ttl

            @local_expiring_lru
            def expensive_meth(self, arg):
                pass
        ..
    """
    @wraps(obj)
    def memoizer(*args, **kwargs):
        instance = args[0]
        lru_size = instance._cache_size
        cache_ttl = instance._cache_ttl
        if lru_size and cache_ttl:
            cache = instance._cache
            kargs = list(args)
            kargs[0] = id(instance)
            key = str((kargs, kwargs))
            try:
                r = list(cache.pop(key))
                if r[1] < datetime.datetime.utcnow():
                    r[0] = None
                else:
                    cache[key] = r
            except (KeyError, AssertionError):
                if len(cache) >= lru_size:
                    cache.popitem(last=False)
                r = cache[key] = (
                    obj(*args, **kwargs),
                    datetime.datetime.utcnow() + datetime.timedelta(
                        seconds=cache_ttl)
                )
            if r[0]:
                return r[0]
        return obj(*args, **kwargs)
    return memoizer