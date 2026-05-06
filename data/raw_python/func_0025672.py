def checkFileExists(filename, directory=None):
    """
    Checks to see if file specified exists in current or specified directory.

    Default is current directory.  Returns 1 if it exists, 0 if not found.
    """

    if directory is not None:
        fname = os.path.join(directory,filename)
    else:
        fname = filename
    _exist = os.path.exists(fname)
    return _exist