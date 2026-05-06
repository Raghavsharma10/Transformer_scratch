def assert_lock(fname):
    """
    If file is locked then terminate program else lock file.
    """

    if not set_lock(fname):
        logger.error('File {} is already locked. Terminating.'.format(fname))
        sys.exit()