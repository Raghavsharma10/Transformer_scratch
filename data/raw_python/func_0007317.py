def patch(destination, name=None, settings=None):
    """Decorator to create a patch.

    The object being decorated becomes the :attr:`~Patch.obj` attribute of the
    patch.

    Parameters
    ----------
    destination : object
        Patch destination.
    name : str
        Name of the attribute at the destination.
    settings : gorilla.Settings
        Settings.

    Returns
    -------
    object
        The decorated object.

    See Also
    --------
    :class:`Patch`.
    """
    def decorator(wrapped):
        base = _get_base(wrapped)
        name_ = base.__name__ if name is None else name
        settings_ = copy.deepcopy(settings)
        patch = Patch(destination, name_, wrapped, settings=settings_)
        data = get_decorator_data(base, set_default=True)
        data.patches.append(patch)
        return wrapped

    return decorator