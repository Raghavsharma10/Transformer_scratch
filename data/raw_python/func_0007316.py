def apply(patch):
    """Apply a patch.

    The patch's :attr:`~Patch.obj` attribute is injected into the patch's
    :attr:`~Patch.destination` under the patch's :attr:`~Patch.name`.

    This is a wrapper around calling
    ``setattr(patch.destination, patch.name, patch.obj)``.

    Parameters
    ----------
    patch : gorilla.Patch
        Patch.

    Raises
    ------
    RuntimeError
        Overwriting an existing attribute is not allowed when the setting
        :attr:`Settings.allow_hit` is set to ``True``.

    Note
    ----
    If both the attributes :attr:`Settings.allow_hit` and
    :attr:`Settings.store_hit` are ``True`` but that the target attribute seems
    to have already been stored, then it won't be stored again to avoid losing
    the original attribute that was stored the first time around.
    """
    settings = Settings() if patch.settings is None else patch.settings

    # When a hit occurs due to an attribute at the destination already existing
    # with the patch's name, the existing attribute is referred to as 'target'.
    try:
        target = get_attribute(patch.destination, patch.name)
    except AttributeError:
        pass
    else:
        if not settings.allow_hit:
            raise RuntimeError(
                "An attribute named '%s' already exists at the destination "
                "'%s'. Set a different name through the patch object to avoid "
                "a name clash or set the setting 'allow_hit' to True to "
                "overwrite the attribute. In the latter case, it is "
                "recommended to also set the 'store_hit' setting to True in "
                "order to store the original attribute under a different "
                "name so it can still be accessed."
                % (patch.name, patch.destination.__name__))

        if settings.store_hit:
            original_name = _ORIGINAL_NAME % (patch.name,)
            if not hasattr(patch.destination, original_name):
                setattr(patch.destination, original_name, target)

    setattr(patch.destination, patch.name, patch.obj)