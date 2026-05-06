def Shift(self, n):
    """
    [
        {
            'self': [1, 2, 3, 4, 5],
            'n': 3,
            'assert': lambda ret: list(ret) == [4, 5, 1, 2, 3]
         }
    ]
    """
    headn = tuple(Take(self, n))
    yield from self
    yield from headn