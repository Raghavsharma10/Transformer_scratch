def split(s, posix=True):
    """Split the string s using shell-like syntax.

    Args:
        s (str): String to split
        posix (bool): Use posix split

    Returns:
        list of str: List of string parts
    """
    if isinstance(s, six.binary_type):
        s = s.decode("utf-8")
    return shlex.split(s, posix=posix)