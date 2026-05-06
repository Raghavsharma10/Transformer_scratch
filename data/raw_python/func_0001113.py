def __rmfile(path):
    """Delete a file.

    Args:
        path (str): Path to the file that needs to be deleted.

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    logger.info("rmfile: %s" % path)
    try:
        os.remove(path)
        return True
    except Exception as e:
        logger.error("rmfile: %s failed! Error: %s" % (path, e))
        return False