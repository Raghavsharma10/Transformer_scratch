def is_float(value, min=None, max=None):
    """
    A check that tests that a given value is a float
    (an integer will be accepted), and optionally - that it is between bounds.

    If the value is a string, then the conversion is done - if possible.
    Otherwise a VdtError is raised.

    This can accept negative values.

    >>> vtor = Validator()
    >>> vtor.check('float', '2')
    2.0

    From now on we multiply the value to avoid comparing decimals

    >>> vtor.check('float', '-6.8') * 10
    -68.0
    >>> vtor.check('float', '12.2') * 10
    122.0
    >>> vtor.check('float', 8.4) * 10
    84.0
    >>> vtor.check('float', 'a')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtTypeError: the value "a" is of the wrong type.
    >>> vtor.check('float(10.1)', '10.2') * 10
    102.0
    >>> vtor.check('float(max=20.2)', '15.1') * 10
    151.0
    >>> vtor.check('float(10.0)', '9.0')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueTooSmallError: the value "9.0" is too small.
    >>> vtor.check('float(max=20.0)', '35.0')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueTooBigError: the value "35.0" is too big.
    """
    (min_val, max_val) = _is_num_param(
        ('min', 'max'), (min, max), to_float=True)
    if not isinstance(value, number_or_string_types):
        raise VdtTypeError(value)
    if not isinstance(value, float):
        # if it's a string - does it represent a float ?
        try:
            value = float(value)
        except ValueError:
            raise VdtTypeError(value)
    if (min_val is not None) and (value < min_val):
        raise VdtValueTooSmallError(value)
    if (max_val is not None) and (value > max_val):
        raise VdtValueTooBigError(value)
    return value