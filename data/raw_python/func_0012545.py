def join_paths(*paths, **kwargs):
    """
    Join multiple paths together and return the absolute path of them. If 'safe'
    is specified, this function will 'clean' the path with the 'safe_path'
    function. This will clean root decelerations from the path
    after the first item.

    Would like to do 'safe=False' instead of '**kwargs' but stupider versions
    of python *cough 2.6* don't like that after '*paths'.

    .. code: python

        reusables.join_paths("var", "\\log", "/test")
        'C:\\Users\\Me\\var\\log\\test'

        os.path.join("var", "\\log", "/test")
        '/test'

    :param paths: paths to join together
    :param kwargs: 'safe', make them into a safe path it True
    :return: abspath as string
    """
    path = os.path.abspath(paths[0])

    for next_path in paths[1:]:
        path = os.path.join(path, next_path.lstrip("\\").lstrip("/").strip())
    path.rstrip(os.sep)
    return path if not kwargs.get('safe') else safe_path(path)