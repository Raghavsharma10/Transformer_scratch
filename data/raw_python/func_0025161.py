def version_is_valid(version_str):
    """ Check to see if the version specified is a valid as far as pkg_resources is concerned

    >>> version_is_valid('blah')
    False
    >>> version_is_valid('1.2.3')
    True
    """
    try:
        packaging.version.Version(version_str)
    except packaging.version.InvalidVersion:
        return False
    return True