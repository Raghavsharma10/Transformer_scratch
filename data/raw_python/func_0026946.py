def TakeIf(self: dict, f):
    """
    [
        {
            'self': [1, 2, 3],
            'f': lambda e: e%2,
            'assert': lambda ret: list(ret)  == [1, 3]
         }
    ]
    """
    if is_to_destruct(f):
        f = destruct_func(f)

    return (e for e in self.items() if f(e))