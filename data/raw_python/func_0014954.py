def first(n, it, constructor=list):
    """
    >>> first(3,iter([1,2,3,4]))
    [1, 2, 3]
    >>> first(3,iter([1,2,3,4]), iter) #doctest: +ELLIPSIS
    <itertools.islice object at ...>
    >>> first(3,iter([1,2,3,4]), tuple)
    (1, 2, 3)
    """
    return constructor(itertools.islice(it,n))