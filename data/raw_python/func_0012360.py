def pickle_load(name, extension='.pkl'):
    """Load data with pickle.

    Parameters
    ----------
    name: str
        Path to save to (includes dir, excludes extension).
    extension: str, optional
        File extension.

    Returns
    -------
    Contents of file path.
    """
    filename = name + extension
    infile = open(filename, 'rb')
    data = pickle.load(infile)
    infile.close()
    return data