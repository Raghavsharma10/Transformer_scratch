def _walk(directory, enable_scandir=False, **kwargs):
    """
    Internal function to return walk generator either from os or scandir

    :param directory: directory to traverse
    :param enable_scandir: on python < 3.5 enable external scandir package
    :param kwargs: arguments to pass to walk function
    :return: walk generator
    """
    walk = os.walk
    if python_version < (3, 5) and enable_scandir:
        import scandir
        walk = scandir.walk
    return walk(directory, **kwargs)