def _use_datastore(self, key, options=None):
    """Return whether to use the datastore for this key.

    Args:
      key: Key instance.
      options: ContextOptions instance, or None.

    Returns:
      True if the datastore should be used, False otherwise.
    """
    flag = ContextOptions.use_datastore(options)
    if flag is None:
      flag = self._datastore_policy(key)
    if flag is None:
      flag = ContextOptions.use_datastore(self._conn.config)
    if flag is None:
      flag = True
    return flag