def list_to_rows(src, size):
    """A generator that takes a enumerable item and returns a series of
    slices. Useful for turning a list into a series of rows. 

    >>> list(list_to_rows([1, 2, 3, 4, 5, 6, 7], 3))
    [[1, 2, 3], [4, 5, 6], [7, ]]
    """

    row = []
    for item in src:
        row.append(item)
        if len(row) == size:
            yield row
            row = []

    if row:
        yield row