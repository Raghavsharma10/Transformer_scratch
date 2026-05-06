def _typed_from_items(items):
    """
    Construct strongly typed attributes (properties) from a dictionary of
    name and :class:`~exa.typed.Typed` object pairs.

    See Also:
        :func:`~exa.typed.typed`
    """
    dct = {}
    for name, attr in items:
        if isinstance(attr, Typed):
            dct[name] = attr(name)
    return dct