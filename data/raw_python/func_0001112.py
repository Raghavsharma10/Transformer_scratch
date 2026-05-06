def gmove(pattern, destination):
    """Move all file found by glob.glob(pattern) to destination directory.

    Args:
        pattern (str): Glob pattern
        destination (str): Path to the destination directory.

    Returns:
        bool: True if the operation is successful, False otherwise.
    """
    for item in glob.glob(pattern):
        if not move(item, destination):
            return False
    return True