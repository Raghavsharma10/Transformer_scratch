def validate_quantity(input_value, output_unit, equivalencies=[]):
    """Validate quantity (value and unit).

    .. note::

        For flux conversion, use :func:`convert_flux` instead.

    Parameters
    ----------
    input_value : number, array-like, or `~astropy.units.quantity.Quantity`
        Quantity to validate. If not a Quantity, assumed to be
        already in output unit.

    output_unit : str or `~astropy.units.core.Unit`
        Output quantity unit.

    equivalencies : list of equivalence pairs, optional
        See `astropy.units`.

    Returns
    -------
    output_value : `~astropy.units.quantity.Quantity`
        Validated quantity in given unit.

    """
    output_unit = validate_unit(output_unit)

    if isinstance(input_value, u.Quantity):
        output_value = input_value.to(output_unit, equivalencies=equivalencies)
    else:
        output_value = input_value * output_unit

    return output_value