def _check_initialized(self):
    """Internal helper to check for uninitialized properties.

    Raises:
      BadValueError if it finds any.
    """
    baddies = self._find_uninitialized()
    if baddies:
      raise datastore_errors.BadValueError(
          'Entity has uninitialized properties: %s' % ', '.join(baddies))