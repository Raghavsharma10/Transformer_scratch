def memoize(func):
    """Memoize a method that should return the same result every time on a
    given instance.

    """
    @wraps(func)
    def memoizer(self):
        if not hasattr(self, '_cache'):
            self._cache = {}
        if func.__name__ not in self._cache:
            self._cache[func.__name__] = func(self)
        return self._cache[func.__name__]
    return memoizer