def parent(self):
    """Return a Key constructed from all but the last (kind, id) pairs.

    If there is only one (kind, id) pair, return None.
    """
    pairs = self.__pairs
    if len(pairs) <= 1:
      return None
    return Key(pairs=pairs[:-1], app=self.__app, namespace=self.__namespace)