def aslist(generator):
    'Function decorator to transform a generator into a list'
    def wrapper(*args, **kwargs):
        return list(generator(*args, **kwargs))
    return wrapper