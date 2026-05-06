def typed(cls):
    """
    Class decorator that updates a class definition with strongly typed
    property attributes.

    See Also:
        If the class will be inherited, use :class:`~exa.typed.TypedClass`.
    """
    for name, attr in _typed_from_items(vars(cls).items()).items():
        setattr(cls, name, attr)
    return cls