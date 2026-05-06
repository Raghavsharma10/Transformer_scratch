def to_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case. For example, "some_var" would become "someVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.lstrip('_').split('_')
    return parts[0] + ''.join([i.title() for i in parts[1:]])