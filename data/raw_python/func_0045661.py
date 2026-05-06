def lines_from_stream(f, as_interned=False):
    """
    Create a list of file lines from a given file stream.

    Args:
        f (io.TextIOWrapper): File stream
        as_interned (bool): List of "interned" strings (default False)

    Returns:
        strings (list): File line list
    """
    if as_interned:
        return [sys.intern(line) for line in f.read().splitlines()]
    return f.read().splitlines()