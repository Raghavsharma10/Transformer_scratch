def read(path, encoding="utf-8"):
    """Read the content of the file.

    Args:
        path (str): Path to the file
        encoding (str): File encoding. Default: utf-8

    Returns:
        str: File content or empty string if there was an error
    """
    try:
        with io.open(path, encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error("read: %s failed. Error: %s", path, e)
        return ""