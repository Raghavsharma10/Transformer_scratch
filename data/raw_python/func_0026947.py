def TakeWhile(self: dict, f):
    """
    [
        {
            'self': [1, 2, 3, 4, 5],
            'f': lambda x: x < 4,
            'assert': lambda ret: list(ret)  == [1, 2, 3]
         }
    ]
    """
    if is_to_destruct(f):
        f = destruct_func(f)

    for e in self.items():
        if not f(e):
            break
        yield e