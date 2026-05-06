def per_chunk(iterable, n=1, fillvalue=None):
    """
    From http://stackoverflow.com/a/8991553/610569

        >>> list(per_chunk('abcdefghi', n=2))
        [('a', 'b'), ('c', 'd'), ('e', 'f'), ('g', 'h'), ('i', None)]
        >>> list(per_chunk('abcdefghi', n=3))
        [('a', 'b', 'c'), ('d', 'e', 'f'), ('g', 'h', 'i')]
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)