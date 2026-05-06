def get_diaginfo(diaginfo_file):
    """
    Read an output's diaginfo.dat file and parse into a DataFrame for
    use in selecting and parsing categories.

    Parameters
    ----------
    diaginfo_file : str
        Path to diaginfo.dat

    Returns
    -------
    DataFrame containing the category information.

    """

    widths = [rec.width for rec in diag_recs]
    col_names = [rec.name for rec in diag_recs]
    dtypes = [rec.type for rec in diag_recs]
    usecols = [name for name in col_names if not name.startswith('-')]

    diag_df = pd.read_fwf(diaginfo_file, widths=widths, names=col_names,
                          dtypes=dtypes, comment="#", header=None,
                          usecols=usecols)
    diag_desc = {diag.name: diag.desc for diag in diag_recs
                 if not diag.name.startswith('-')}

    return diag_df, diag_desc