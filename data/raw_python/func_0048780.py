def make_temp(suffix="", prefix="tmp", dir=None):
    """
    Creates a temporary file with a closed stream and deletes it when done.

    :return: A contextmanager retrieving the file path.
    """
    temporary = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=dir)
    os.close(temporary[0])
    try:
        yield temporary[1]
    finally:
        os.remove(temporary[1])