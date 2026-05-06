def _use_memcache(self, key, options=None):
    """Return whether to use memcache for this key.

    Args:
      key: Key instance.
      options: ContextOptions instance, or None.

    Returns:
      True if the key should be cached in memcache, False otherwise.
    """
    flag = ContextOptions.use_memcache(options)
    if flag is None:
      flag = self._memcache_policy(key)
    if flag is None:
      flag = ContextOptions.use_memcache(self._conn.config)
    if flag is None:
      flag = True
    return flag