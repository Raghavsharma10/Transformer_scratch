def load_precision(filename):
    """
    Load a CLASS precision file into a dictionary.

    Parameters
    ----------
    filename : str
        the name of an existing file to load, or one in the files included
        as part of the CLASS source

    Returns
    -------
    dict :
        the precision parameters loaded from file
    """
    # also look in data dir
    path = _find_file(filename)

    r = dict()
    with open(path, 'r') as f:
        exec(f.read(), {}, r)

    return r