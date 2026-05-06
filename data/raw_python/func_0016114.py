def assert_valid_path(path):
    """Checks if a path is a correct format that Marathon expects. Raises ValueError if not valid.

    :param str path: The app id.

    :rtype: str
    """
    if path is None:
        return
    # As seen in:
    # https://github.com/mesosphere/marathon/blob/0c11661ca2f259f8a903d114ef79023649a6f04b/src/main/scala/mesosphere/marathon/state/PathId.scala#L71
    for id in filter(None, path.strip('/').split('/')):
        if not ID_PATTERN.match(id):
            raise ValueError(
                'invalid path (allowed: lowercase letters, digits, hyphen, "/", ".", ".."): %r' % path)
    return path