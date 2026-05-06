def flat(self):
    """Return a tuple of alternating kind and id values."""
    flat = []
    for kind, id in self.__pairs:
      flat.append(kind)
      flat.append(id)
    return tuple(flat)