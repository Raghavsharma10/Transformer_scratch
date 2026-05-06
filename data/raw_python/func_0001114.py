def __rmtree(path):
    """Recursively delete a directory tree.

    Args:
        path (str): Path to the directory that needs to be deleted.

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    logger.info("rmtree: %s" % path)
    try:
        shutil.rmtree(path)
        return True
    except Exception as e:
        logger.error("rmtree: %s failed! Error: %s" % (path, e))
        return False