def Drop(self: Iterable, n):
    """
    [
        {
            'self': [1, 2, 3, 4, 5],
            'n': 3,
            'assert': lambda ret: list(ret) == [1, 2]
         }
    ]
    """
    con = tuple(self)
    n = len(con) - n
    if n <= 0:
        yield from con
    else:
        for i, e in enumerate(con):
            if i == n:
                break
            yield e