def string_id(self):
    """Return the string id in the last (kind, id) pair, if any.

    Returns:
      A string id, or None if the key has an integer id or is incomplete.
    """
    id = self.id()
    if not isinstance(id, basestring):
      id = None
    return id