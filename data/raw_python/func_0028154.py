def kwargs_to_variable_assignment(kwargs: dict, value_representation=repr,
                                  assignment_operator: str = ' = ',
                                  statement_separator: str = '\n',
                                  statement_per_line: bool = False) -> str:
    """
    Convert a dictionary into a string with assignments

    Each assignment is constructed based on:
    key assignment_operator value_representation(value) statement_separator,
    where key and value are the key and value of the dictionary.
    Moreover one can seprate the assignment statements by new lines.

    Parameters
    ----------
    kwargs : dict

    assignment_operator: str, optional:
        Assignment operator (" = " in python)
    value_representation: str, optinal
        How to represent the value in the assignments (repr function in python)
    statement_separator : str, optional:
        Statement separator (new line in python)
    statement_per_line: bool, optional
        Insert each statement on a different line

    Returns
    -------
    str
        All the assignemnts.

    >>> kwargs_to_variable_assignment({'a': 2, 'b': "abc"})
    "a = 2\\nb = 'abc'\\n"
    >>> kwargs_to_variable_assignment({'a':2 ,'b': "abc"}, statement_per_line=True)
    "a = 2\\n\\nb = 'abc'\\n"
    >>> kwargs_to_variable_assignment({'a': 2})
    'a = 2\\n'
    >>> kwargs_to_variable_assignment({'a': 2}, statement_per_line=True)
    'a = 2\\n'
    """
    code = []
    join_str = '\n' if statement_per_line else ''
    for key, value in kwargs.items():
        code.append(key + assignment_operator +
                    value_representation(value)+statement_separator)
    return join_str.join(code)