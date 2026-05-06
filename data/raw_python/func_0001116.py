def gremove(pattern):
    """Remove all file found by glob.glob(pattern).

    Args:
        pattern (str): Pattern of files to remove
    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    for item in glob.glob(pattern):
        if not remove(item):
            return False
    return True