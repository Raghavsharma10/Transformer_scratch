def mkdir(path, mode=0o755, delete=False):
    """Make a directory.

    Create a leaf directory and all intermediate ones.
    Works like ``mkdir``, except that any intermediate path segment (not just
    the rightmost) will be created if it does not exist. This is recursive.

    Args:
        path (str): Directory to create
        mode (int): Directory mode
        delete (bool): Delete directory/file if exists

    Returns:
        bool: True if succeeded else False
    """
    logger.info("mkdir: %s" % path)
    if os.path.isdir(path):
        if not delete:
            return True
        if not remove(path):
            return False
    try:
        os.makedirs(path, mode)
        return True
    except Exception:
        logger.exception("Failed to mkdir: %s" % path)
        return False