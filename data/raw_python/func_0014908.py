def binarySearchItem(seq, item, cmpfunc=cmp):
    r""" Search an ordered sequence `seq` for `item`, using comparison function
    `cmpfunc` (defaults to ``cmp``) and return the first found instance of
    `item`, or `None` if item is not in `seq`. The returned item is NOT
    guaranteed to be the first occurrence of item in `seq`."""
    pos = binarySearchPos(seq, item, cmpfunc)
    if pos == -1: raise KeyError("Item not in seq")
    else:         return seq[pos]