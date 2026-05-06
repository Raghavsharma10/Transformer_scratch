def _find_uninitialized(self):
    """Internal helper to find uninitialized properties.

    Returns:
      A set of property names.
    """
    return set(name
               for name, prop in self._properties.iteritems()
               if not prop._is_initialized(self))