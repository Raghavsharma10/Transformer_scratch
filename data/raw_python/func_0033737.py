def smart_round( val, decimal_places = 2 ):
    """
    For floats >= 10.**-(decimal_places - 1), rounds off to the valber of decimal places specified.
    For floats < 10.**-(decimal_places - 1), puts in exponential form then rounds off to the decimal
    places specified.
    @val: value to round; if val is not a float, just returns val
    @decimal_places: number of decimal places to round to
    """
    if isinstance(val, float) and val != 0.0:
        if val >= 10.**-(decimal_places - 1):
            conv_str = ''.join([ '%.', str(decimal_places), 'f' ])
        else:
            conv_str = ''.join([ '%.', str(decimal_places), 'e' ])
        val = float( conv_str % val )

    return val