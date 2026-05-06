def set_cache_policy(self, func):
    """Set the context cache policy function.

    Args:
      func: A function that accepts a Key instance as argument and returns
        a bool indicating if it should be cached.  May be None.
    """
    if func is None:
      func = self.default_cache_policy
    elif isinstance(func, bool):
      func = lambda unused_key, flag=func: flag
    self._cache_policy = func