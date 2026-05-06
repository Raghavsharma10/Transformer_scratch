def patches(destination, settings=None, traverse_bases=True,
            filter=default_filter, recursive=True, use_decorators=True):
    """Decorator to create a patch for each member of a module or a class.

    Parameters
    ----------
    destination : object
        Patch destination.
    settings : gorilla.Settings
        Settings.
    traverse_bases : bool
        If the object is a class, the base classes are also traversed.
    filter : function
        Attributes for which the function returns ``False`` are skipped. The
        function needs to define two parameters: ``name``, the attribute name,
        and ``obj``, the attribute value. If ``None``, no attribute is skipped.
    recursive : bool
        If ``True``, and a hit occurs due to an attribute at the destination
        already existing with the given name, and both the member and the
        target attributes are classes, then instead of creating a patch
        directly with the member attribute value as is, a patch for each of its
        own members is created with the target as new destination.
    use_decorators : bool
        Allows to take any modifier decorator into consideration to allow for
        more granular customizations.

    Returns
    -------
    object
        The decorated object.

    Note
    ----
    A 'target' differs from a 'destination' in that a target represents an
    existing attribute at the destination about to be hit by a patch.

    See Also
    --------
    :class:`Patch`, :func:`create_patches`.
    """
    def decorator(wrapped):
        settings_ = copy.deepcopy(settings)
        patches = create_patches(
            destination, wrapped, settings=settings_,
            traverse_bases=traverse_bases, filter=filter, recursive=recursive,
            use_decorators=use_decorators)
        data = get_decorator_data(_get_base(wrapped), set_default=True)
        data.patches.extend(patches)
        return wrapped

    return decorator