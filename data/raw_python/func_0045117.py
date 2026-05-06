def mkp(*args, **kwargs):
    """
    Generate a directory path, and create it if requested.

    .. code-block:: Python

        filepath = mkp('base', 'folder', 'file')
        dirpath = mkp('root', 'path', 'folder', mk=True)

    Args:
        \*args: File or directory path segments to be concatenated
        mk (bool): Make the directory (if it doesn't exist)

    Returns:
        path (str): File or directory path
    """
    mk = kwargs.pop('mk', False)
    path = os.sep.join(list(args))
    if mk:
        while sep2 in path:
            path = path.replace(sep2, os.sep)
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    return path