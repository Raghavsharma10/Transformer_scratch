def key_for_property(cls, kind, property):
    """Return the __property__ key for property of kind.

    Args:
      kind: kind whose key is requested.
      property: property whose key is requested.

    Returns:
      The key for property of kind.
    """
    return model.Key(Kind.KIND_NAME, kind, Property.KIND_NAME, property)