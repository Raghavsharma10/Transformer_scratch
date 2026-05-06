def force_list(value, min=None, max=None):
    """
    Check that a value is a list, coercing strings into
    a list with one member. Useful where users forget the
    trailing comma that turns a single value into a list.

    You can optionally specify the minimum and maximum number of members.
    A minumum of greater than one will fail if the user only supplies a
    string.

    >>> vtor = Validator()
    >>> vtor.check('force_list', ())
    []
    >>> vtor.check('force_list', [])
    []
    >>> vtor.check('force_list', 'hello')
    ['hello']
    """
    if not isinstance(value, (list, tuple)):
        value = [value]
    return is_list(value, min, max)