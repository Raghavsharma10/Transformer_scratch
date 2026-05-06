def memoize(function):
    """Memoizing function.  Potentially not thread-safe, since it will return
    resuts across threads.  Make sure this is okay with callers."""
    _cache = {}
    @wraps(function)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in _cache:
            _cache[key] = function(*args, **kwargs)
        return _cache[key]
    return wrapper