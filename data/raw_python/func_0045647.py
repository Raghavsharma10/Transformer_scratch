def _mkdirs(d):
    """
    Make all directories up to d.

    No exception is raised if d exists.
    """
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise