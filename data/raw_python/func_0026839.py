def Take(self: Iterable, n):
    """
    [
        {
            'self': [1, 2, 3],
            'n': 2,
            'assert': lambda ret: list(ret)  == [1, 2]
         }
    ]
    """

    for i, e in enumerate(self):
        if i == n:
            break
        yield e