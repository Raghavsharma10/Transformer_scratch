def vector_to_symmetric(v):
    '''Convert an iterable into a symmetric matrix.'''
    np = len(v)
    N = (int(sqrt(1 + 8*np)) - 1)//2
    if N*(N+1)//2 != np:
        raise ValueError('Cannot convert vector to symmetric matrix')
    sym = ndarray((N,N))
    iterable = iter(v)
    for r in range(N):
        for c in range(r+1):
            sym[r,c] = sym[c,r] = iterable.next()
    return sym