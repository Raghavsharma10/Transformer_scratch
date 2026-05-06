def check_path(path):
    """Check that a path is legal.

    :return: the path if all is OK
    :raise ValueError: if the path is illegal
    """
    if path is None or path == b'' or path.startswith(b'/'):
        raise ValueError("illegal path '%s'" % path)

    if (
        (sys.version_info[0] >= 3 and not isinstance(path, bytes)) and
        (sys.version_info[0] == 2 and not isinstance(path, str))
    ):
        raise TypeError("illegale type for path '%r'" % path)

    return path