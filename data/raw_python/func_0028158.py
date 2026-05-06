def find_duplicates(l: list) -> set:
    """
    Return the duplicates in a list.

    The function relies on
    https://stackoverflow.com/questions/9835762/find-and-list-duplicates-in-a-list .
    Parameters
    ----------
    l : list
        Name

    Returns
    -------
    set
        Duplicated values

    >>> find_duplicates([1,2,3])
    set()
    >>> find_duplicates([1,2,1])
    {1}
    """
    return set([x for x in l if l.count(x) > 1])