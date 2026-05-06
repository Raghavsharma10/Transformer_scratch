def is_ip_addr(value):
    """
    Check that the supplied value is an Internet Protocol address, v.4,
    represented by a dotted-quad string, i.e. '1.2.3.4'.

    >>> vtor = Validator()
    >>> vtor.check('ip_addr', '1 ')
    '1'
    >>> vtor.check('ip_addr', ' 1.2')
    '1.2'
    >>> vtor.check('ip_addr', ' 1.2.3 ')
    '1.2.3'
    >>> vtor.check('ip_addr', '1.2.3.4')
    '1.2.3.4'
    >>> vtor.check('ip_addr', '0.0.0.0')
    '0.0.0.0'
    >>> vtor.check('ip_addr', '255.255.255.255')
    '255.255.255.255'
    >>> vtor.check('ip_addr', '255.255.255.256')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueError: the value "255.255.255.256" is unacceptable.
    >>> vtor.check('ip_addr', '1.2.3.4.5')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueError: the value "1.2.3.4.5" is unacceptable.
    >>> vtor.check('ip_addr', 0)  # doctest: +SKIP
    Traceback (most recent call last):
    VdtTypeError: the value "0" is of the wrong type.
    """
    if not isinstance(value, string_types):
        raise VdtTypeError(value)
    value = value.strip()
    try:
        dottedQuadToNum(value)
    except ValueError:
        raise VdtValueError(value)
    return value