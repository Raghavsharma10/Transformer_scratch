def copy(source, destination):
    """Copy file or directory.

    Args:
        source (str): Source file or directory
        destination (str): Destination file or directory (where to copy).

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    if os.path.isdir(source):
        return __copytree(source, destination)
    else:
        return __copyfile2(source, destination)