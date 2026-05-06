def assert_not_exists(path, sep='.'):
    """
    If path exists, modify to add a counter in the filename. Useful
    for preventing accidental overrides. For example, if `file.txt`
    exists, check if `file.1.txt` also exists. Repeat until we find
    a non-existing version, such as `file.12.txt`.

    Parameters
    ----------
    path : str
        Path to be checked

    Returns
    -------
    newpath : str
        A modified version of path with a counter right before the extension.
    """
    name, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(path):
        path = '{}{}{}{}'.format(name, sep, i, ext)
        i += 1
    return path