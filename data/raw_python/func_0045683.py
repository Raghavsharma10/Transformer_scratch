def type_to_desc(type_: Type) -> str:
    """Convert a type to a human-readable description.

    This is used for generating nice error messages. We want users \
    to see a nice readable text, rather than something like \
    "typing.List<~T>[str]".

    Args:
        type_: The type to represent.

    Returns:
        A human-readable description.
    """
    scalar_type_to_str = {
        str: 'string',
        int: 'int',
        float: 'float',
        bool: 'boolean',
        None: 'null value',
        type(None): 'null value'
    }

    if type_ in scalar_type_to_str:
        return scalar_type_to_str[type_]

    if is_generic_union(type_):
        return 'union of {}'.format([type_to_desc(t)
                                     for t in generic_type_args(type_)])

    if is_generic_list(type_):
        return 'list of ({})'.format(type_to_desc(generic_type_args(type_)[0]))

    if is_generic_dict(type_):
        return 'dict of string to ({})'.format(
                type_to_desc(generic_type_args(type_)[1]))

    return type_.__name__