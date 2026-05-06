def get_file_size(path):
    """The the size of a file in bytes.

    Parameters
    ----------
    path: str
        The path of the file.

    Returns
    -------
    int
        The size of the file in bytes.

    Raises
    ------
    IOError
        If the file does not exist.
    OSError
        If a file system error occurs.
    """
    assert isinstance(path, (str, _oldstr))

    if not os.path.isfile(path):
        raise IOError('File "%s" does not exist.', path)

    return os.path.getsize(path)