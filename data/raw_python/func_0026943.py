def ChunkBy(self: dict, f=None):
    """
    [
        {
            'self': [1, 1, 3, 3, 1, 1],
            'f': lambda x: x%2,
            'assert': lambda ret: ret == [[1, 1], [3, 3], [1, 1]]
         }
    ]
    """
    if f is None:
        return _chunk(self.items())
    if is_to_destruct(f):
        f = destruct_func(f)
    return _chunk(self.items(), f)