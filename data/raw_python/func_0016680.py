def argshash(self, args, kwargs):
        "Hash mutable arguments' containers with immutable keys and values."
        a = repr(args)
        b = repr(sorted((repr(k), repr(v)) for k, v in kwargs.items()))
        return a + b