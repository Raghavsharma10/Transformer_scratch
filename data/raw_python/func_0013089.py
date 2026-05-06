def _get_or_insert(*args, **kwds):
    """Transactionally retrieves an existing entity or creates a new one.

    Positional Args:
      name: Key name to retrieve or create.

    Keyword Args:
      namespace: Optional namespace.
      app: Optional app ID.
      parent: Parent entity key, if any.
      context_options: ContextOptions object (not keyword args!) or None.
      **kwds: Keyword arguments to pass to the constructor of the model class
        if an instance for the specified key name does not already exist. If
        an instance with the supplied key_name and parent already exists,
        these arguments will be discarded.

    Returns:
      Existing instance of Model class with the specified key name and parent
      or a new one that has just been created.
    """
    cls, args = args[0], args[1:]
    return cls._get_or_insert_async(*args, **kwds).get_result()