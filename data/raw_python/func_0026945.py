def Take(self: dict, n):
    """
    [
        {
            'self': [1, 2, 3],
            'n': 2,
            'assert': lambda ret: list(ret)  == [1, 2]
         }
    ]
    """

    for i, e in enumerate(self.items()):
        if i == n:
            break
        yield e