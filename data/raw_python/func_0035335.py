def distance(a, b):
    """Compute tonnetz-distance between two chromagrams.
    
    ----
    >>> C = np.zeros(12)
    >>> C[0] = 1
    >>> D = np.zeros(12)
    >>> D[2] = 1
    >>> G = np.zeros(12)
    >>> G[7] = 1

    The distance is zero on equivalent chords
    >>> distance(C, C) == 0
    True

    The distance is symetric
    >>> distance(C, D) == distance(D, C)
    True

    >>> distance(C, D) > 0
    True
    >>> distance(C, G) < distance(C, D)
    True
    """
    [a_tonnetz, b_tonnetz] = [_to_tonnetz(x) for x in [a, b]]
    return np.linalg.norm(b_tonnetz - a_tonnetz)