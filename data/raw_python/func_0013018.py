def _load_from_cache_if_available(self, key):
    """Returns a cached Model instance given the entity key if available.

    Args:
      key: Key instance.

    Returns:
      A Model instance if the key exists in the cache.
    """
    if key in self._cache:
      entity = self._cache[key]  # May be None, meaning "doesn't exist".
      if entity is None or entity._key == key:
        # If entity's key didn't change later, it is ok.
        # See issue 13.  http://goo.gl/jxjOP
        raise tasklets.Return(entity)