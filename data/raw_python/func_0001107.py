def __copyfile2(source, destination):
    """Copy data and all stat info ("cp -p source destination").

    The destination may be a directory.

    Args:
        source (str): Source file (file to copy).
        destination (str): Destination file or directory (where to copy).

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    logger.info("copyfile2: %s -> %s" % (source, destination))
    try:
        __create_destdir(destination)
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        logger.error(
            "copyfile2: %s -> %s failed! Error: %s", source, destination, e
        )
        return False