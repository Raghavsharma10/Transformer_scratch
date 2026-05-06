def open(pattern, read_only=False):
    """
    Return a root descriptor to work with one or multiple NetCDF files.

    Keyword arguments:
    pattern -- a list of filenames or a string pattern.
    """
    root = NCObject.open(pattern, read_only=read_only)
    return root, root.is_new