def version_diff(version1, version2):
    """Return string representing the diff between package versions.

    We're interested in whether this is a major, minor, patch or 'other'
    update. This method will compare the two versions and return None if
    they are the same, else it will return a string value indicating the
    type of diff - 'major', 'minor', 'patch', 'other'.

    Args:
        version1: the Version object we are interested in (e.g. current)
        version2: the Version object to compare against (e.g. latest)

    Returns a string - 'major', 'minor', 'patch', 'other', or None if the
        two are identical.

    """
    if version1 is None or version2 is None:
        return 'unknown'
    if version1 == version2:
        return 'none'

    for v in ('major', 'minor', 'patch'):
        if getattr(version1, v) != getattr(version2, v):
            return v

    return 'other'