def is_list(value, min=None, max=None):
    """
    Check that the value is a list of values.

    You can optionally specify the minimum and maximum number of members.

    It does no check on list members.

    >>> vtor = Validator()
    >>> vtor.check('list', ())
    []
    >>> vtor.check('list', [])
    []
    >>> vtor.check('list', (1, 2))
    [1, 2]
    >>> vtor.check('list', [1, 2])
    [1, 2]
    >>> vtor.check('list(3)', (1, 2))  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueTooShortError: the value "(1, 2)" is too short.
    >>> vtor.check('list(max=5)', (1, 2, 3, 4, 5, 6))  # doctest: +SKIP
    Traceback (most recent call last):
    VdtValueTooLongError: the value "(1, 2, 3, 4, 5, 6)" is too long.
    >>> vtor.check('list(min=3, max=5)', (1, 2, 3, 4))  # doctest: +SKIP
    [1, 2, 3, 4]
    >>> vtor.check('list', 0)  # doctest: +SKIP
    Traceback (most recent call last):
    VdtTypeError: the value "0" is of the wrong type.
    >>> vtor.check('list', '12')  # doctest: +SKIP
    Traceback (most recent call last):
    VdtTypeError: the value "12" is of the wrong type.
    """
    (min_len, max_len) = _is_num_param(('min', 'max'), (min, max))
    if isinstance(value, string_types):
        raise VdtTypeError(value)
    try:
        num_members = len(value)
    except TypeError:
        raise VdtTypeError(value)
    if min_len is not None and num_members < min_len:
        raise VdtValueTooShortError(value)
    if max_len is not None and num_members > max_len:
        raise VdtValueTooLongError(value)
    return list(value)