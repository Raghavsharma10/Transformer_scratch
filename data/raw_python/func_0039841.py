def isdir(path, message):
    """
    Raise an exception if the given directory does not exist.

    :param path: The path to a directory to be tested
    :param message: A custom message to report in the exception

    :raises: FileNotFoundError
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(
            errno.ENOENT,
            "{}: {}".format(message, os.strerror(errno.ENOENT)), path)