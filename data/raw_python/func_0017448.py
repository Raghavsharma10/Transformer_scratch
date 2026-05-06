def make_skiplist(*args, use_fallback=False):
    '''Create a new skiplist'''
    sl = fallback.Skiplist if use_fallback else Skiplist
    return sl(*args)