def __copyfile(source, destination):
    """Copy data and mode bits ("cp source destination").

    The destination may be a directory.

    Args:
        source (str): Source file (file to copy).
        destination (str): Destination file or directory (where to copy).

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    logger.info("copyfile: %s -> %s" % (source, destination))
    try:
        __create_destdir(destination)
        shutil.copy(source, destination)
        return True
    except Exception as e:
        logger.error(
            "copyfile: %s -> %s failed! Error: %s", source, destination, e
        )
        return False