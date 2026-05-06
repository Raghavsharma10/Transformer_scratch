def clean_caches(path):
    """
    Removes all python cache files recursively on a path.

    :param path: the path
    :return: None
    """

    for dirname, subdirlist, filelist in os.walk(path):

        for f in filelist:
            if f.endswith('pyc'):
                try:
                    os.remove(os.path.join(dirname, f))
                except FileNotFoundError:
                    pass

        if dirname.endswith('__pycache__'):
            shutil.rmtree(dirname)