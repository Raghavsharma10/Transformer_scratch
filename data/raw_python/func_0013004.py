def key_to_kind(cls, key):
    """Return the kind specified by a given __property__ key.

    Args:
      key: key whose kind name is requested.

    Returns:
      The kind specified by key.
    """
    if key.kind() == Kind.KIND_NAME:
      return key.id()
    else:
      return key.parent().id()