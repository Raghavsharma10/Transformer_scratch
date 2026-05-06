def touch(path, content="", encoding="utf-8", overwrite=False):
    """Create a file at the given path if it does not already exists.

    Args:
        path (str): Path to the file.
        content (str): Optional content that will be written in the file.
        encoding (str): Encoding in which to write the content.
            Default: ``utf-8``
        overwrite (bool): Overwrite the file if exists.

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    path = os.path.abspath(path)
    if not overwrite and os.path.exists(path):
        logger.warning('touch: "%s" already exists', path)
        return False
    try:
        logger.info("touch: %s", path)
        with io.open(path, "wb") as f:
            if not isinstance(content, six.binary_type):
                content = content.encode(encoding)
            f.write(content)
        return True
    except Exception as e:
        logger.error("touch: %s failed. Error: %s", path, e)
        return False