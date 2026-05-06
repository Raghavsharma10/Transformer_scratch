def filter(value):
    """Modifier decorator to force the inclusion or exclusion of an attribute.

    This only modifies the behaviour of the :func:`create_patches` function
    and the :func:`patches` decorator, given that their parameter
    ``use_decorators`` is set to ``True``.

    Parameters
    ----------
    value : bool
        ``True`` to force inclusion, ``False`` to force exclusion, and ``None``
        to inherit from the behaviour defined by :func:`create_patches` or
        :func:`patches`.

    Returns
    -------
    object
        The decorated object.
    """
    def decorator(wrapped):
        data = get_decorator_data(_get_base(wrapped), set_default=True)
        data.filter = value
        return wrapped

    return decorator