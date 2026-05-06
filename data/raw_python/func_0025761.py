def is_integer(value, min=None, max=None):
    """
    A check that tests that a given value is an integer (int, or long)
    and optionally, between bounds. A negative value is accepted, while
    a float will fail.

    If the value is a string, then the conversion is done - if possible.
    Otherwise a VdtError is raised.

    >>> vtor = Validator()
    >>> vtor.check('integer', '-1')
    -1
    >>> vtor.check('integer', '0')
    0
    >>> vtor.check('integer', 9)
    9
    >>> vtor.check('integer', 'a')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtTypeError: the value "a" is of the wrong type.
    >>> vtor.check('integer', '2.2')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtTypeError: the value "2.2" is of the wrong type.
    >>> vtor.check('integer(10)', '20')
    20
    >>> vtor.check('integer(max=20)', '15')
    15
    >>> vtor.check('integer(10)', '9')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueTooSmallError: the value "9" is too small.
    >>> vtor.check('integer(10)', 9)  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueTooSmallError: the value "9" is too small.
    >>> vtor.check('integer(max=20)', '35')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueTooBigError: the value "35" is too big.
    >>> vtor.check('integer(max=20)', 35)  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueTooBigError: the value "35" is too big.
    >>> vtor.check('integer(0, 9)', False)
    0
    """
    (min_val, max_val) = _is_num_param(('min', 'max'), (min, max))
    if not isinstance(value, int_or_string_types):
        raise VdtTypeError(value)
    if isinstance(value, string_types):
        # if it's a string - does it represent an integer ?
        try:
            value = int(value)
        except ValueError:
            raise VdtTypeError(value)
    if (min_val is not None) and (value < min_val):
        raise VdtValueTooSmallError(value)
    if (max_val is not None) and (value > max_val):
        raise VdtValueTooBigError(value)
    return value