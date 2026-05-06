def create_patches(destination, root, settings=None, traverse_bases=True,
                   filter=default_filter, recursive=True, use_decorators=True):
    """Create a patch for each member of a module or a class.

    Parameters
    ----------
    destination : object
        Patch destination.
    root : object
        Root object, either a module or a class.
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
        ``True`` to take any modifier decorator into consideration to allow for
        more granular customizations.

    Returns
    -------
    list of gorilla.Patch
        The patches.

    Note
    ----
    A 'target' differs from a 'destination' in that a target represents an
    existing attribute at the destination about to be hit by a patch.

    See Also
    --------
    :func:`patches`.
    """
    if filter is None:
        filter = _true

    out = []
    root_patch = Patch(destination, '', root, settings=settings)
    stack = collections.deque((root_patch,))
    while stack:
        parent_patch = stack.popleft()
        members = _get_members(parent_patch.obj, traverse_bases=traverse_bases,
                               filter=None, recursive=False)
        for name, value in members:
            patch = Patch(parent_patch.destination, name, value,
                          settings=copy.deepcopy(parent_patch.settings))
            if use_decorators:
                base = _get_base(value)
                decorator_data = get_decorator_data(base)
                filter_override = (None if decorator_data is None
                                   else decorator_data.filter)
                if ((filter_override is None and not filter(name, value))
                        or filter_override is False):
                    continue

                if decorator_data is not None:
                    patch._update(**decorator_data.override)
            elif not filter(name, value):
                continue

            if recursive and isinstance(value, _CLASS_TYPES):
                try:
                    target = get_attribute(patch.destination, patch.name)
                except AttributeError:
                    pass
                else:
                    if isinstance(target, _CLASS_TYPES):
                        patch.destination = target
                        stack.append(patch)
                        continue

            out.append(patch)

    return out