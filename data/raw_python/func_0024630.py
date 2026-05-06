def validate_unit(input_unit):
    """Validate unit.

    To be compatible with existing SYNPHOT data files:

        * 'angstroms' and 'inversemicrons' are accepted although
          unrecognized by astropy units
        * 'transmission', 'extinction', and 'emissivity' are
          converted to astropy dimensionless unit

    Parameters
    ----------
    input_unit : str or `~astropy.units.core.Unit`
        Unit to validate.

    Returns
    -------
    output_unit : `~astropy.units.core.Unit`
        Validated unit.

    Raises
    ------
    synphot.exceptions.SynphotError
        Invalid unit.

    """
    if isinstance(input_unit, str):
        input_unit_lowcase = input_unit.lower()

        # Backward-compatibility
        if input_unit_lowcase == 'angstroms':
            output_unit = u.AA
        elif input_unit_lowcase == 'inversemicrons':
            output_unit = u.micron ** -1
        elif input_unit_lowcase in ('transmission', 'extinction',
                                    'emissivity'):
            output_unit = THROUGHPUT
        elif input_unit_lowcase == 'jy':
            output_unit = u.Jy

        # Work around mag unit limitations
        elif input_unit_lowcase in ('stmag', 'mag(st)'):
            output_unit = u.STmag
        elif input_unit_lowcase in ('abmag', 'mag(ab)'):
            output_unit = u.ABmag

        else:
            try:  # astropy.units is case-sensitive
                output_unit = u.Unit(input_unit)
            except ValueError:  # synphot is case-insensitive
                output_unit = u.Unit(input_unit_lowcase)

    elif isinstance(input_unit, (u.UnitBase, u.LogUnit)):
        output_unit = input_unit

    else:
        raise exceptions.SynphotError(
            '{0} must be a recognized string or '
            'astropy.units.core.Unit'.format(input_unit))

    return output_unit