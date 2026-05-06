def bump(component='patch', exact=None):
    # type: (str, str) -> Tuple[str, str]
    """ Bump the given version component.

    Args:
        component (str):
            What part of the version should be bumped. Can be one of:

            - major
            - minor
            - patch

        exact (str):
            The exact version that should be set instead of bumping the current
            one.

    Returns:
        tuple(str, str): A tuple of old and bumped version.
    """
    old_ver = current()

    if exact is None:
        new_ver = _bump_version(old_ver, component)
    else:
        new_ver = exact

    write(new_ver)
    return old_ver, new_ver