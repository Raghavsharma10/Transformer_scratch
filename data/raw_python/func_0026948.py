def Drop(self: dict, n):
    """
    [
        {
            'self': [1, 2, 3, 4, 5],
            'n': 3,
            'assert': lambda ret: list(ret) == [1, 2]
         }
    ]
    """
    n = len(self) - n
    if n <= 0:
        yield from self.items()
    else:
        for i, e in enumerate(self.items()):
            if i == n:
                break
            yield e