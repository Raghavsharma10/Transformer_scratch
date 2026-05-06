def _izip(*iterables):
    """ Iterate through multiple lists or arrays of equal size """
    # This izip routine is from itertools
    # izip('ABCD', 'xy') --> Ax By

    iterators = map(iter, iterables)
    while iterators:
        yield tuple(map(next, iterators))