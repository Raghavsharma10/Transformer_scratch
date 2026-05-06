def write_table(table, filename):
    """
    Write a table to a file.

    Parameters
    ----------
    table : Table
        Table to be written

    filename : str
        Destination for saving table.

    Returns
    -------
    None
    """
    try:
        if os.path.exists(filename):
            os.remove(filename)
        table.write(filename)
        log.info("Wrote {0}".format(filename))
    except Exception as e:
        if "Format could not be identified" not in e.message:
            raise e
        else:
            fmt = os.path.splitext(filename)[-1][1:].lower()  # extension sans '.'
            raise Exception("Cannot auto-determine format for {0}".format(fmt))
    return