def proj_path(*path_parts):
    # type: (str) -> str
    """ Return absolute path to the repo dir (root project directory).

    Args:
        path (str):
            The path relative to the project root (pelconf.yaml).

    Returns:
        str: The given path converted to an absolute path.
    """
    path_parts = path_parts or ['.']

    # If path represented by path_parts is absolute, do not modify it.
    if not os.path.isabs(path_parts[0]):
        proj_path = _find_proj_root()

        if proj_path is not None:
            path_parts = [proj_path] + list(path_parts)

    return os.path.normpath(os.path.join(*path_parts))