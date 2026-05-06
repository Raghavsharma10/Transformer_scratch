def check_handle_syntax(string):
    '''
    Checks the syntax of a handle without an index (are prefix
    and suffix there, are there too many slashes?).

    :string: The handle without index, as string prefix/suffix.
    :raise: :exc:`~b2handle.handleexceptions.handleexceptions.HandleSyntaxError`
    :return: True. If it's not ok, exceptions are raised.

    '''

    expected = 'prefix/suffix'

    try:
        arr = string.split('/')
    except AttributeError:
        raise handleexceptions.HandleSyntaxError(msg='The provided handle is None', expected_syntax=expected)

    if len(arr) < 2:
        msg = 'No slash'
        raise handleexceptions.HandleSyntaxError(msg=msg, handle=string, expected_syntax=expected)

    if len(arr[0]) == 0:
        msg = 'Empty prefix'
        raise handleexceptions.HandleSyntaxError(msg=msg, handle=string, expected_syntax=expected)

    if len(arr[1]) == 0:
        msg = 'Empty suffix'
        raise handleexceptions.HandleSyntaxError(msg=msg, handle=string, expected_syntax=expected)

    if ':' in string:
        check_handle_syntax_with_index(string, base_already_checked=True)

    return True