def __copytree(source, destination, symlinks=False):
    """Copy a directory tree recursively using copy2().

    The destination directory must not already exist.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied.

    Args:
        source (str): Source directory (directory to copy).
        destination (str): Destination directory (where to copy).
        symlinks (bool): Follow symbolic links.

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    logger.info("copytree: %s -> %s" % (source, destination))
    try:
        __create_destdir(destination)
        shutil.copytree(source, destination, symlinks)
        return True
    except Exception as e:
        logger.exception(
            "copytree: %s -> %s failed! Error: %s", source, destination, e
        )
        return False