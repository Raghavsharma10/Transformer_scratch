def valid_variable_identifier(string):
    """Raises an |ValueError| if the given name is not a valid Python
    identifier.

    For example, the string `test_1` (with underscore) is valid...

    >>> from hydpy.core.objecttools import valid_variable_identifier
    >>> valid_variable_identifier('test_1')

    ...but the string `test 1` (with white space) is not:

    >>> valid_variable_identifier('test 1')
    Traceback (most recent call last):
    ...
    ValueError: The given name string `test 1` does not define a valid \
variable identifier.  Valid identifiers do not contain characters like \
`-` or empty spaces, do not start with numbers, cannot be mistaken with \
Python built-ins like `for`...)

    Also, names of Python built ins are not allowed:

    >>> valid_variable_identifier('print')   # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The given name string `print` does not define...
    """
    string = str(string)
    try:
        exec('%s = None' % string)
        if string in dir(builtins):
            raise SyntaxError()
    except SyntaxError:
        raise ValueError(
            'The given name string `%s` does not define a valid variable '
            'identifier.  Valid identifiers do not contain characters like '
            '`-` or empty spaces, do not start with numbers, cannot be '
            'mistaken with Python built-ins like `for`...)'
            % string)