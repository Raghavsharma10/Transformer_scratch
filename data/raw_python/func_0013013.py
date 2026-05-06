def set_datastore_policy(self, func):
    """Set the context datastore policy function.

    Args:
      func: A function that accepts a Key instance as argument and returns
        a bool indicating if it should use the datastore.  May be None.
    """
    if func is None:
      func = self.default_datastore_policy
    elif isinstance(func, bool):
      func = lambda unused_key, flag=func: flag
    self._datastore_policy = func