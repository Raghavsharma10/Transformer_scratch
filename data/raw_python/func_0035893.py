def pretty_print_head(dict_, count=10): #TODO only format and rename to pretty_head
    '''
    Pretty print some items of a dict.

    For an unordered dict, ``count`` arbitrary items will be printed.

    Parameters
    ----------
    dict_ : ~typing.Dict
        Dict to print from.
    count : int
        Number of items to print.

    Raises
    ------
    ValueError
        When ``count < 1``.
    '''
    if count < 1:
        raise ValueError('`count` must be at least 1')
    pprint(dict(take(count, dict_.items())))