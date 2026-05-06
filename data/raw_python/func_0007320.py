def settings(**kwargs):
    """Modifier decorator to update a patch's settings.

    This only modifies the behaviour of the :func:`create_patches` function
    and the :func:`patches` decorator, given that their parameter
    ``use_decorators`` is set to ``True``.

    Parameters
    ----------
    kwargs
        Settings to update. See :class:`Settings` for the list.

    Returns
    -------
    object
        The decorated object.
    """
    def decorator(wrapped):
        data = get_decorator_data(_get_base(wrapped), set_default=True)
        data.override.setdefault('settings', {}).update(kwargs)
        return wrapped

    return decorator