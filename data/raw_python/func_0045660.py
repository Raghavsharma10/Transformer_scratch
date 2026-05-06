def lines_from_file(path, as_interned=False, encoding=None):
    """
    Create a list of file lines from a given filepath.

    Args:
        path (str): File path
        as_interned (bool): List of "interned" strings (default False)

    Returns:
        strings (list): File line list
    """
    lines = None
    with io.open(path, encoding=encoding) as f:
        if as_interned:
            lines = [sys.intern(line) for line in f.read().splitlines()]
        else:
            lines = f.read().splitlines()
    return lines