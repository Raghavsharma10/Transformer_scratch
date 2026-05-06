def get_cfcompliant_units(units, prefix='', suffix=''):
    """
    Get equivalent units that are compatible with the udunits2 library
    (thus CF-compliant).

    Parameters
    ----------
    units : string
        A string representation of the units.
    prefix : string
        Will be added at the beginning of the returned string
        (must be a valid udunits2 expression).
    suffix : string
        Will be added at the end of the returned string
        (must be a valid udunits2 expression).

    Returns
    -------
    A string representation of the conforming units.

    References
    ----------
    The udunits2 package : http://www.unidata.ucar.edu/software/udunits/

    Notes
    -----
    This function only relies on the table stored in :attr:`UNITS_MAP_CTM2CF`.
    Therefore, the units string returned by this function is not certified to
    be compatible with udunits2.

    Examples
    --------
    >>> get_cfcompliant_units('molec/cm2')
    'count/cm2'
    >>> get_cfcompliant_units('v/v')
    '1'
    >>> get_cfcompliant_units('ppbC', prefix='3')
    '3ppb

    """
    compliant_units = units

    for gcunits, udunits in UNITS_MAP_CTM2CF:
        compliant_units = str.replace(compliant_units, gcunits, udunits)

    return prefix + compliant_units + suffix