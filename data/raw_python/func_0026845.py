def Concat(self: Iterable, *others):
    """
    [
        {
            'self': [1, 2, 3],
            ':args': [[4, 5, 6], [7, 8, 9]],
            'assert': lambda ret: list(ret) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
         }
    ]
    """
    return concat_generator(self, *[unbox_if_flow(other) for other in others])