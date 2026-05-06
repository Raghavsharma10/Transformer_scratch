def iflatten(seq, isSeq=isSeq):
    r"""Like `flatten` but lazy."""
    for elt in seq:
        if isSeq(elt):
            for x in iflatten(elt, isSeq):
                yield x
        else:
            yield elt