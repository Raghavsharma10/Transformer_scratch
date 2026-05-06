def set_memcache_timeout_policy(self, func):
    """Set the policy function for memcache timeout (expiration).

    Args:
      func: A function that accepts a key instance as argument and returns
        an integer indicating the desired memcache timeout.  May be None.

    If the function returns 0 it implies the default timeout.
    """
    if func is None:
      func = self.default_memcache_timeout_policy
    elif isinstance(func, (int, long)):
      func = lambda unused_key, flag=func: flag
    self._memcache_timeout_policy = func