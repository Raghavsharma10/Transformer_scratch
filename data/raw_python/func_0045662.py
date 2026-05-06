def lines_from_string(string, as_interned=False):
    """
    Create a list of file lines from a given string.

    Args:
        string (str): File string
        as_interned (bool): List of "interned" strings (default False)

    Returns:
        strings (list): File line list
    """
    if as_interned:
        return [sys.intern(line) for line in string.splitlines()]
    return string.splitlines()