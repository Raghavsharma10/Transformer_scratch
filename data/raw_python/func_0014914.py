def weave(*iterables):
    r"""weave(seq1 [, seq2] [...]) ->  iter([seq1[0], seq2[0] ...]).

    >>> list(weave([1,2,3], [4,5,6,'A'], [6,7,8, 'B', 'C']))
    [1, 4, 6, 2, 5, 7, 3, 6, 8]

    Any iterable will work. The first exhausted iterable determines when to
    stop. FIXME rethink stopping semantics.

    >>> list(weave(iter(('is','psu')), ('there','no', 'censorship')))
    ['is', 'there', 'psu', 'no']
    >>> list(weave(('there','no', 'censorship'), iter(('is','psu'))))
    ['there', 'is', 'no', 'psu', 'censorship']
    """
    iterables = map(iter, iterables)
    while True:
        for it in iterables: yield it.next()