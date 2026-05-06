def invert(dict_): #TODO return a MultiDict right away
    '''
    Invert dict by swapping each value with its key.

    Parameters
    ----------
    dict_ : ~typing.Dict[~typing.Hashable, ~typing.Hashable]
        Dict to invert.

    Returns
    -------
    ~typing.Dict[~typing.Hashable, ~typing.Set[~typing.Hashable]]
        Dict with keys and values swapped.

    See also
    --------
    pytil.multi_dict.MultiDict : Multi-dict view of a ``Dict[Hashable, Set[Hashable]]`` dict.

    Notes
    -----
    If your dict never has 2 keys mapped to the same value, you can convert it
    to a ``Dict[Hashable, Hashable]`` dict using::

        from pytil.multi_dict import MultiDict
        inverted_dict = dict(MultiDict(inverted_dict))

    Examples
    --------
    >>> invert({1: 2, 3: 4})
    {2: {1}, 4: {3}}

    >>> invert({1: 2, 3: 2, 4: 5})
    {2: {1,3}, 5: {4}}
    '''
    result = defaultdict(lambda: set())
    for k, val in dict_.items():
        result[val].add(k)
    return dict(result)