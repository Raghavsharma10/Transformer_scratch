def show_formats():
    """
    Print a list of all the file formats that are supported for writing.
    The file formats are determined by their extensions.

    Returns
    -------
    None
    """
    fmts = {
        "ann": "Kvis annotation",
        "reg": "DS9 regions file",
        "fits": "FITS Binary Table",
        "csv": "Comma separated values",
        "tab": "tabe separated values",
        "tex": "LaTeX table format",
        "html": "HTML table",
        "vot": "VO-Table",
        "xml": "VO-Table",
        "db": "Sqlite3 database",
        "sqlite": "Sqlite3 database"}
    supported = get_table_formats()
    print("Extension |     Description       | Supported?")
    for k in sorted(fmts.keys()):
        print("{0:10s} {1:24s} {2}".format(k, fmts[k], k in supported))
    return