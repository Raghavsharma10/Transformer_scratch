def Skip(self: dict, n):
    """
        [
            {
                'self': [1, 2, 3, 4, 5],
                'n': 3,
                'assert': lambda ret: list(ret) == [4, 5]
             }
        ]
        """

    con = self.items()
    for i, _ in enumerate(con):
        if i == n:
            break
    return con