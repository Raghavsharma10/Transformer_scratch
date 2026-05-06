def writeFITSTable(filename, table):
    """
    Convert a table into a FITSTable and then write to disk.

    Parameters
    ----------
    filename : str
        Filename to write.

    table : Table
        Table to write.

    Returns
    -------
    None

    Notes
    -----
    Due to a bug in numpy, `int32` and `float32` are converted to `int64` and `float64` before writing.
    """
    def FITSTableType(val):
        """
        Return the FITSTable type corresponding to each named parameter in obj
        """
        if isinstance(val, bool):
            types = "L"
        elif isinstance(val, (int, np.int64, np.int32)):
            types = "J"
        elif isinstance(val, (float, np.float64, np.float32)):
            types = "E"
        elif isinstance(val, six.string_types):
            types = "{0}A".format(len(val))
        else:
            log.warning("Column {0} is of unknown type {1}".format(val, type(val)))
            log.warning("Using 5A")
            types = "5A"
        return types

    cols = []
    for name in table.colnames:
        cols.append(fits.Column(name=name, format=FITSTableType(table[name][0]), array=table[name]))
    cols = fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(cols)
    for k in table.meta:
        tbhdu.header['HISTORY'] = ':'.join((k, table.meta[k]))
    tbhdu.writeto(filename, overwrite=True)