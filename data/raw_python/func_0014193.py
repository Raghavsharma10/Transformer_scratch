def flatten(*args):
    '''Generator that recursively flattens embedded lists, tuples, etc.'''
    for arg in args:
        if isinstance(arg, collections.Iterable) and not isinstance(arg, (str, bytes)):
            yield from flatten(*arg)
        else:
            yield arg