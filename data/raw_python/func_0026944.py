def GroupBy(self: dict, f=None):
    """
    [
        {
            'self': [1, 2, 3],
            'f': lambda x: x%2,
            'assert': lambda ret: ret[0] == [2] and ret[1] == [1, 3]
         }
    ]
    """
    if f and is_to_destruct(f):
        f = destruct_func(f)
    return _group_by(self.items(), f)