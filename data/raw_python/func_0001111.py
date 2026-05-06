def move(source, destination):
    """Move a file or directory (recursively) to another location.

    If the destination is on our current file system, then simply use
    rename. Otherwise, copy source to the destination and then remove
    source.

    Args:
        source (str): Source file or directory (file or directory to move).
        destination (str): Destination file or directory (where to move).

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    logger.info("Move: %s -> %s" % (source, destination))
    try:
        __create_destdir(destination)
        shutil.move(source, destination)
        return True
    except Exception:
        logger.exception("Failed to Move: %s -> %s" % (source, destination))
        return False