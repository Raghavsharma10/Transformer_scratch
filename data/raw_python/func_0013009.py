def _use_cache(self, key, options=None):
    """Return whether to use the context cache for this key.

    Args:
      key: Key instance.
      options: ContextOptions instance, or None.

    Returns:
      True if the key should be cached, False otherwise.
    """
    flag = ContextOptions.use_cache(options)
    if flag is None:
      flag = self._cache_policy(key)
    if flag is None:
      flag = ContextOptions.use_cache(self._conn.config)
    if flag is None:
      flag = True
    return flag