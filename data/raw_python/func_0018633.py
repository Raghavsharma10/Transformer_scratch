def convert_pint_to_fortran_safe_units(units, inverse=False):
    """
    Convert Pint units to Fortran safe units

    Parameters
    ----------
    units : list_like, str
        Units to convert

    inverse : bool
        If True, convert the other way i.e. convert Fortran safe units to Pint units

    Returns
    -------
    ``type(units)``
        Set of converted units
    """
    if inverse:
        return apply_string_substitutions(units, FORTRAN_SAFE_TO_PINT_UNITS_MAPPING)
    else:
        return apply_string_substitutions(units, PINT_TO_FORTRAN_SAFE_UNITS_MAPPING)