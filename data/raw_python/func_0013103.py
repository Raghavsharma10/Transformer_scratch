def integer_id(self):
    """Return the integer id in the last (kind, id) pair, if any.

    Returns:
      An integer id, or None if the key has a string id or is incomplete.
    """
    id = self.id()
    if not isinstance(id, (int, long)):
      id = None
    return id