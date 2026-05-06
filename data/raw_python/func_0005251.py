def unique_list(seq):
    """ Removes duplicate elements from given @seq

        @seq: a #list or sequence-like object

        -> #list
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]