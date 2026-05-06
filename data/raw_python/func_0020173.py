def Chunks(l, n, all=False):
    '''
    Returns a generator of consecutive `n`-sized chunks of list `l`.
    If `all` is `True`, returns **all** `n`-sized chunks in `l`
    by iterating over the starting point.

    '''

    if all:
        jarr = range(0, n - 1)
    else:
        jarr = [0]

    for j in jarr:
        for i in range(j, len(l), n):
            if i + 2 * n <= len(l):
                yield l[i:i + n]
            else:
                if not all:
                    yield l[i:]
                break