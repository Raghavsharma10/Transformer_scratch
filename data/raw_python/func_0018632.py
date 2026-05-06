def get_pint_to_fortran_safe_units_mapping(inverse=False):
    """Get the mappings from Pint to Fortran safe units.

    Fortran can't handle special characters like "^" or "/" in names, but we need
    these in Pint. Conversely, Pint stores variables with spaces by default e.g. "Mt
    CO2 / yr" but we don't want these in the input files as Fortran is likely to think
    the whitespace is a delimiter.

    Parameters
    ----------
    inverse : bool
        If True, return the inverse mappings i.e. Fortran safe to Pint mappings

    Returns
    -------
    dict
        Dictionary of mappings
    """
    replacements = {"^": "super", "/": "per", " ": ""}
    if inverse:
        replacements = {v: k for k, v in replacements.items()}
        # mapping nothing to something is obviously not going to work in the inverse
        # hence remove
        replacements.pop("")

    return replacements