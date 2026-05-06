def _radixPass(a, b, r, n, K):
    """
    Stable sort of the sequence a according to the keys given in r.

    >>> a=range(5)
    >>> b=[0]*5
    >>> r=[2,1,3,0,4]

    >>> _radixPass(a, b, r, 5, 5)
    >>> b
    [3, 1, 0, 2, 4]

    When n is less than the length of a, the end of b must be left unaltered.
    >>> b=[5]*5
    >>> _radixPass(a, b, r, 2, 2)
    >>> b
    [1, 0, 5, 5, 5]

    >>> _a=a=[1, 0]
    >>> b= [0]*2
    >>> r=[0, 1]
    >>> _radixPass(a, b, r, 2, 2)
    >>> a=_a
    >>> b
    [0, 1]

    >>> a=[1, 1]
    >>> _radixPass(a, b, r, 2, 2)
    >>> b
    [1, 1]

    >>> a=[0, 1, 1, 0]
    >>> b= [0]*4
    >>> r=[0, 1]
    >>> _radixPass(a, b, r, 4, 2)
    >>> a=_a
    >>> b
    [0, 0, 1, 1]
    """
    c = _array("i", [0] * (K + 1))  # counter array

    for i in range(n):  # count occurrences
        c[r[a[i]]] += 1

    sum = 0

    for i in range(K + 1):  # exclusive prefix sums
        t = c[i]
        c[i] = sum
        sum += t

    for a_i in a[:n]:  # sort
        b[c[r[a_i]]] = a_i
        c[r[a_i]] += 1