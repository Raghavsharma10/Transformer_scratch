def parseFilename(filename):
    """
    Parse out filename from any specified extensions.

    Returns rootname and string version of extension name.
    """

    # Parse out any extension specified in filename
    _indx = filename.find('[')
    if _indx > 0:
        # Read extension name provided
        _fname = filename[:_indx]
        _extn = filename[_indx + 1:-1]
    else:
        _fname = filename
        _extn = None

    return _fname, _extn