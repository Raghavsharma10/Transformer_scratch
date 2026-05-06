def variations(iterable, optional=lambda x: False):
    """ Returns all possible variations of a sequence with optional items.
    """
    # For example: variations(["A?", "B?", "C"], optional=lambda s: s.endswith("?"))
    # defines a sequence where constraint A and B are optional:
    # [("A?", "B?", "C"), ("B?", "C"), ("A?", "C"), ("C")]
    iterable = tuple(iterable)
    # Create a boolean sequence where True means optional:
    # ("A?", "B?", "C") => [True, True, False]
    o = [optional(x) for x in iterable]
    # Find all permutations of the boolean sequence:
    # [True, False, True], [True, False, False], [False, False, True], [False, False, False].
    # Map to sequences of constraints whose index in the boolean sequence yields True.
    a = set()
    for p in product([False, True], repeat=sum(o)):
        p = list(p)
        v = [b and (b and p.pop(0)) for b in o]
        v = tuple(iterable[i] for i in xrange(len(v)) if not v[i])
        a.add(v)
    # Longest-first.
    return sorted(a, cmp=lambda x, y: len(y) - len(x))