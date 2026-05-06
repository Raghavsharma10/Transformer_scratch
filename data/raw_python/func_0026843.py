def Skip(self: Iterable, n):
    """
        [
            {
                'self': [1, 2, 3, 4, 5],
                'n': 3,
                'assert': lambda ret: list(ret) == [4, 5]
             }
        ]
        """
    con = iter(self)
    for i, _ in enumerate(con):
        if i == n:
            break
    return con