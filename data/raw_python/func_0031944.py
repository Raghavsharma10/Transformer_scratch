def normalize_directory(path):
    """
    Append "/" to `path` if needed.
    """
    if path is None:
        return None
    if path.endswith(os.path.sep):
        return path
    else:
        return path + os.path.sep