def load_table(filename):
    """
    Load a table from a given file.

    Supports csv, tab, tex, vo, vot, xml, fits, and hdf5.

    Parameters
    ----------
    filename : str
        File to read

    Returns
    -------
    table : Table
        Table of data.
    """
    supported = get_table_formats()

    fmt = os.path.splitext(filename)[-1][1:].lower()  # extension sans '.'

    if fmt in ['csv', 'tab', 'tex'] and fmt in supported:
        log.info("Reading file {0}".format(filename))
        t = ascii.read(filename)
    elif fmt in ['vo', 'vot', 'xml', 'fits', 'hdf5'] and fmt in supported:
        log.info("Reading file {0}".format(filename))
        t = Table.read(filename)
    else:
        log.error("Table format not recognized or supported")
        log.error("{0} [{1}]".format(filename, fmt))
        raise Exception("Table format not recognized or supported")
    return t