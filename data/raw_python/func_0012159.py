def quote_str(obj):
    r"""
    Add extra quotes to a string.

    If the argument is not a string it is returned unmodified.

    :param obj: Object
    :type  obj: any

    :rtype: Same as argument

    For example:

        >>> import pmisc
        >>> pmisc.quote_str(5)
        5
        >>> pmisc.quote_str('Hello!')
        '"Hello!"'
        >>> pmisc.quote_str('He said "hello!"')
        '\'He said "hello!"\''
    """
    if not isinstance(obj, str):
        return obj
    return "'{obj}'".format(obj=obj) if '"' in obj else '"{obj}"'.format(obj=obj)