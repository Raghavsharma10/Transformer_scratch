def get_valid_varname(varname):
    """
    Replace characters (e.g., ':', '$', '=', '-') of a variable name, which
    may cause problems when using with (CF-)netCDF based packages.

    Parameters
    ----------
    varname : string
        variable name.

    Notes
    -----
    Characters replacement is based on the table stored in
    :attr:`VARNAME_MAP_CHAR`.

    """
    vname = varname
    for s, r in VARNAME_MAP_CHAR:
        vname = vname.replace(s, r)

    return vname