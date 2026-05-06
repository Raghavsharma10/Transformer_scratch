def join_path_prefix(path, pre_path=None):
    """
    If path set and not absolute, append it to pre path (if used)

    :param path: path to append
    :type path: str | None
    :param pre_path: Base path to append to (default: None)
    :type pre_path: None | str
    :return: Path or appended path
    :rtype: str | None
    """
    if not path:
        return path

    if pre_path and not os.path.isabs(path):
        return os.path.join(pre_path, path)

    return path