def to_capitalized_camel_case(snake_case_string):
    """
    Convert a string from snake case to camel case with the first letter capitalized. For example, "some_var"
    would become "SomeVar".

    :param snake_case_string: Snake-cased string to convert to camel case.
    :returns: Camel-cased version of snake_case_string.
    """
    parts = snake_case_string.split('_')
    return ''.join([i.title() for i in parts])