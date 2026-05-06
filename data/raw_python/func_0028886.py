def to_snake_case(camel_case_string):
    """
    Convert a string from camel case to snake case. From example, "someVar" would become "some_var".

    :param camel_case_string: Camel-cased string to convert to snake case.
    :return: Snake-cased version of camel_case_string.
    """
    first_pass = _first_camel_case_regex.sub(r'\1_\2', camel_case_string)
    return _second_camel_case_regex.sub(r'\1_\2', first_pass).lower()