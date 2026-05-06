def remove(path):
    """Delete a file or directory.

    Args:
        path (str): Path to the file or directory that needs to be deleted.

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    if os.path.isdir(path):
        return __rmtree(path)
    else:
        return __rmfile(path)