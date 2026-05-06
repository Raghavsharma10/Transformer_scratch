def camel_case(snake_str):
    """
    Returns a camel-cased version of a string.

    :param a_string: any :class:`str` object.

    Usage:
        >>> camel_case('foo_bar')
        "fooBar"
    """

    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])