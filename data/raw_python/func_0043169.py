def store_integers(items, allow_zero=True):
    """Store integers from the given list in a storage.

    This is an example function to show autodoc style.

    Return :class:`Storage` instance with integers from the given list.

    Examples::

        >>> storage = store_integers([1, 'foo', 2, 'bar', 0])
        >>> storage.items
        [1, 2, 0]
        >>> storage = store_integers([1, 'foo', 2, 'bar', 0], allow_zero=False)
        >>> storage.items
        [1, 2]

    :param items:
        List of objects of any type, only :class:`int` instances will be
        stored.
    :param allow_zero:
        Boolean -- if ``False``, ``0`` integers will be skipped.
        Defaults to ``True``.

    """
    ints = [x for x in items if isinstance(x, int) and (allow_zero or x != 0)]
    storage = Storage(ints)
    return storage