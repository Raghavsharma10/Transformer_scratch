def memcache_get(self, key, for_cas=False, namespace=None, use_cache=False,
                   deadline=None):
    """An auto-batching wrapper for memcache.get() or .get_multi().

    Args:
      key: Key to set.  This must be a string; no prefix is applied.
      for_cas: If True, request and store CAS ids on the Context.
      namespace: Optional namespace.
      deadline: Optional deadline for the RPC.

    Returns:
      A Future (!) whose return value is the value retrieved from
      memcache, or None.
    """
    if not isinstance(key, basestring):
      raise TypeError('key must be a string; received %r' % key)
    if not isinstance(for_cas, bool):
      raise TypeError('for_cas must be a bool; received %r' % for_cas)
    if namespace is None:
      namespace = namespace_manager.get_namespace()
    options = (for_cas, namespace, deadline)
    batcher = self.memcache_get_batcher
    if use_cache:
      return batcher.add_once(key, options)
    else:
      return batcher.add(key, options)