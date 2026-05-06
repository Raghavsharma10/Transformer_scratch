def _bump_version(version, component='patch'):
    # type: (str, str) -> str
    """ Bump the given version component.

    Args:
        version (str):
            The current version. The format is: MAJOR.MINOR[.PATCH].
        component (str):
            What part of the version should be bumped. Can be one of:

            - major
            - minor
            - patch

    Returns:
        str: Bumped version as a string.
    """
    if component not in ('major', 'minor', 'patch'):
        raise ValueError("Invalid version component: {}".format(component))

    m = RE_VERSION.match(version)
    if m is None:
        raise ValueError("Version must be in MAJOR.MINOR[.PATCH] format")

    major = m.group('major')
    minor = m.group('minor') or '0'
    patch = m.group('patch') or None

    if patch == '0':
        patch = None

    if component == 'major':
        major = str(int(major) + 1)
        minor = '0'
        patch = None

    elif component == 'minor':
        minor = str(int(minor) + 1)
        patch = None

    else:
        patch = patch or 0
        patch = str(int(patch) + 1)

    new_ver = '{}.{}'.format(major, minor)
    if patch is not None:
        new_ver += '.' + patch

    return new_ver