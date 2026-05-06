def _find_file(filename):
    """
    Find the file path, first checking if it exists and then looking in the
    data directory
    """
    import os
    if os.path.exists(filename):
        path = filename
    else:
        path = os.path.dirname(__file__)
        path = os.path.join(path, 'data', filename)

    if not os.path.exists(path):
        raise ValueError("cannot locate file '%s'" %filename)

    return path