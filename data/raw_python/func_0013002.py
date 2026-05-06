def key_for_namespace(cls, namespace):
    """Return the Key for a namespace.

    Args:
      namespace: A string giving the namespace whose key is requested.

    Returns:
      The Key for the namespace.
    """
    if namespace:
      return model.Key(cls.KIND_NAME, namespace)
    else:
      return model.Key(cls.KIND_NAME, cls.EMPTY_NAMESPACE_ID)