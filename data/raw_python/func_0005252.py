def flatten(*seqs):
    """ Flattens a sequence e.g. |[(1, 2), (3, (4, 5))] -> [1, 2, 3, 4, 5]|

        @seq: #tuple, #list or :class:UserList

        -> yields an iterator

        ..
            l = [(1, 2), (3, 4)]
            for x in flatten(l):
                print(x)
        ..
    """
    for seq in seqs:
        for item in seq:
            if isinstance(item, (tuple, list, UserList)):
                for subitem in flatten(item):
                    yield subitem
            else:
                yield item