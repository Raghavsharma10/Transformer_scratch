def _get_by_id(cls, id, parent=None, **ctx_options):
    """Returns an instance of Model class by ID.

    This is really just a shorthand for Key(cls, id, ...).get().

    Args:
      id: A string or integer key ID.
      parent: Optional parent key of the model to get.
      namespace: Optional namespace.
      app: Optional app ID.
      **ctx_options: Context options.

    Returns:
      A model instance or None if not found.
    """
    return cls._get_by_id_async(id, parent=parent, **ctx_options).get_result()