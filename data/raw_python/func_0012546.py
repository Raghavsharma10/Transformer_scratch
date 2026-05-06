def join_here(*paths, **kwargs):
    """
    Join any path or paths as a sub directory of the current file's directory.

    .. code:: python

        reusables.join_here("Makefile")
        # 'C:\\Reusables\\Makefile'

    :param paths: paths to join together
    :param kwargs: 'strict', do not strip os.sep
    :param kwargs: 'safe', make them into a safe path it True
    :return: abspath as string
    """
    path = os.path.abspath(".")
    for next_path in paths:
        next_path = next_path.lstrip("\\").lstrip("/").strip() if not \
            kwargs.get('strict') else next_path
        path = os.path.abspath(os.path.join(path, next_path))
    return path if not kwargs.get('safe') else safe_path(path)