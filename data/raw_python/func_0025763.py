def is_boolean(value):
    """
    Check if the value represents a boolean.

    >>> vtor = Validator()
    >>> vtor.check('boolean', 0)
    0
    >>> vtor.check('boolean', False)
    0
    >>> vtor.check('boolean', '0')
    0
    >>> vtor.check('boolean', 'off')
    0
    >>> vtor.check('boolean', 'false')
    0
    >>> vtor.check('boolean', 'no')
    0
    >>> vtor.check('boolean', 'nO')
    0
    >>> vtor.check('boolean', 'NO')
    0
    >>> vtor.check('boolean', 1)
    1
    >>> vtor.check('boolean', True)
    1
    >>> vtor.check('boolean', '1')
    1
    >>> vtor.check('boolean', 'on')
    1
    >>> vtor.check('boolean', 'true')
    1
    >>> vtor.check('boolean', 'yes')
    1
    >>> vtor.check('boolean', 'Yes')
    1
    >>> vtor.check('boolean', 'YES')
    1
    >>> vtor.check('boolean', '')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtTypeError: the value "" is of the wrong type.
    >>> vtor.check('boolean', 'up')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtTypeError: the value "up" is of the wrong type.

    """
    if isinstance(value, string_types):
        try:
            return bool_dict[value.lower()]
        except KeyError:
            raise VdtTypeError(value)
    # we do an equality test rather than an identity test
    # this ensures Python 2.2 compatibilty
    # and allows 0 and 1 to represent True and False
    if value == False:
        return False
    elif value == True:
        return True
    else:
        raise VdtTypeError(value)