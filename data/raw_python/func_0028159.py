def sort_dict(d: dict, by: str = 'key',
              allow_duplicates: bool = True) -> collections.OrderedDict:
    """
    Sort a dictionary by key or value.

    The function relies on
    https://docs.python.org/3/library/collections.html#collections.OrderedDict .
    The dulicated are determined based on
    https://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-a-list .
    Parameters
    ----------
    d : dict
        Input dictionary
    by : ['key','value'], optional
        By what to sort the input dictionary
    allow_duplicates : bool, optional
        Flag to indicate if the duplicates are allowed.
    Returns
    -------
    collections.OrderedDict
        Sorted dictionary.

    >>> sort_dict({2: 3, 1: 2, 3: 1})
    OrderedDict([(1, 2), (2, 3), (3, 1)])
    >>> sort_dict({2: 3, 1: 2, 3: 1}, by='value')
    OrderedDict([(3, 1), (1, 2), (2, 3)])
    >>> sort_dict({'2': 3, '1': 2}, by='value')
    OrderedDict([('1', 2), ('2', 3)])
    >>> sort_dict({2: 1, 1: 2, 3: 1}, by='value', allow_duplicates=False)
    Traceback (most recent call last):
        ...
    ValueError: There are duplicates in the values: {1}
    >>> sort_dict({1:1,2:3},by=True)
    Traceback (most recent call last):
        ...
    ValueError: by can be 'key' or 'value'.
    """
    if by == 'key':
        i = 0
    elif by == 'value':
        values = list(d.values())
        if len(values) != len(set(values)) and not allow_duplicates:
            duplicates = find_duplicates(values)
            raise ValueError("There are duplicates in the values: {}".format(duplicates))
        i = 1
    else:
        raise ValueError("by can be 'key' or 'value'.")

    return collections.OrderedDict(sorted(d.items(), key=lambda t: t[i]))