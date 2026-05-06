def safe_make_dirs(path, mode=0o777):
    """Create a leaf directory and all intermediate ones in a safe way.

    A wrapper to os.makedirs() that handles existing leaf directories while
    avoiding os.path.exists() race conditions.

    :param path: relative or absolute directory tree to create
    :param mode: directory permissions in octal
    :return: The newly-created path
    """
    try:
        os.makedirs(path, mode)
    except OSError as error:
        if error.errno != 17:  # File exists
            raise

    return path