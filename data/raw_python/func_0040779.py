def check_file_existence(path):
    """
    :return: FileType
    :rtype: int
    :raises InvalidFilePathError:
    :raises FileNotFoundError:
    :raises RuntimeError:
    """

    pathvalidate.validate_file_path(path)

    if not os.path.lexists(path):
        raise FileNotFoundError(path)

    if os.path.isfile(path):
        logger.debug("file found: " + path)
        return FileType.FILE

    if os.path.isdir(path):
        logger.debug("directory found: " + path)
        return FileType.DIRECTORY

    if os.path.islink(path):
        logger.debug("link found: " + path)
        return FileType.LINK

    raise RuntimeError()