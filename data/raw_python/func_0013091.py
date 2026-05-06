def _allocate_ids(cls, size=None, max=None, parent=None, **ctx_options):
    """Allocates a range of key IDs for this model class.

    Args:
      size: Number of IDs to allocate. Either size or max can be specified,
        not both.
      max: Maximum ID to allocate. Either size or max can be specified,
        not both.
      parent: Parent key for which the IDs will be allocated.
      **ctx_options: Context options.

    Returns:
      A tuple with (start, end) for the allocated range, inclusive.
    """
    return cls._allocate_ids_async(size=size, max=max, parent=parent,
                                   **ctx_options).get_result()