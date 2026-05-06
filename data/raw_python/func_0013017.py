def _get_memcache_timeout(self, key, options=None):
    """Return the memcache timeout (expiration) for this key."""
    timeout = ContextOptions.memcache_timeout(options)
    if timeout is None:
      timeout = self._memcache_timeout_policy(key)
    if timeout is None:
      timeout = ContextOptions.memcache_timeout(self._conn.config)
    if timeout is None:
      timeout = 0
    return timeout