def cached_property(prop):
    """
    A replacement for the property decorator that will only compute the
    attribute's value on the first call and serve a cached copy from
    then on.
    """
    def cache_wrapper(self):
        if not hasattr(self, "_cache"):
            self._cache = {}
        if prop.__name__ not in self._cache:
            return_value = prop(self)
            if isgenerator(return_value):
                return_value = tuple(return_value)
            self._cache[prop.__name__] = return_value
        return self._cache[prop.__name__]
    return property(cache_wrapper)