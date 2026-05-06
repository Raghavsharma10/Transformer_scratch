def make_dummy_tuples(chars='abcde'):
    """
    Helper function to create a list of namedtuples which are useful
    for testing and debugging (especially of custom generators).

    Example
    -------
    >>> make_dummy_tuples(chars='abcd')
    [Quux(x='AA', y='aa'),
     Quux(x='BB', y='bb'),
     Quux(x='CC', y='cc'),
     Quux(x='DD', y='dd')]
    """
    Quux = namedtuple('Quux', ['x', 'y'])
    some_tuples = [Quux((c*2).upper(), c*2) for c in chars]
    return some_tuples