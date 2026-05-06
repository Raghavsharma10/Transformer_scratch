def is_option(value, *options):
    """
    This check matches the value to any of a set of options.

    >>> vtor = Validator()
    >>> vtor.check('option("yoda", "jedi")', 'yoda')
    'yoda'
    >>> vtor.check('option("yoda", "jedi")', 'jed')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueError: the value "jed" is unacceptable.
    >>> vtor.check('option("yoda", "jedi")', 0)  # doctest: +SKIP
    Traceback (most recent call last):
    VdtTypeError: the value "0" is of the wrong type.
    """
    if not isinstance(value, string_types):
        raise VdtTypeError(value)
    if not value in options:
        raise VdtValueError(value)
    return value