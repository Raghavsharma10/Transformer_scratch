def check_table_formats(files):
    """
    Determine whether a list of files are of a recognizable output type.

    Parameters
    ----------
    files : str
        A list of file names

    Returns
    -------
    result : bool
        True if *all* the file names are supported
    """
    cont = True
    formats = get_table_formats()
    for t in files.split(','):
        _, ext = os.path.splitext(t)
        ext = ext[1:].lower()
        if ext not in formats:
            cont = False
            log.warn("Format not supported for {0} ({1})".format(t, ext))
    if not cont:
        log.error("Invalid table format specified.")
    return cont