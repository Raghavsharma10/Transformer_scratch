def memo(f):
    "Return a function like f that remembers and reuses results of past calls."
    table = {}
    def memo_f(*args):
        try:
            return table[args]
        except KeyError:
            table[args] = value = f(*args)
            return value
    return memo_f