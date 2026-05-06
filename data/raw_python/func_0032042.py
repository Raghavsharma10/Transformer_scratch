def _longest_common_subsequence(x, y):
    """
    Return the longest common subsequence between two sequences.

    Parameters
    ----------
    x, y : sequence

    Returns
    -------
    sequence
        Longest common subsequence of x and y.

    Examples
    --------
    >>> _longest_common_subsequence("AGGTAB", "GXTXAYB")
    ['G', 'T', 'A', 'B']
    >>> _longest_common_subsequence(["A", "GA", "G", "T", "A", "B"],
    ...                             ["GA", "X", "T", "X", "A", "Y", "B"])
    ['GA', 'T', 'A', 'B']

    """
    m = len(x)
    n = len(y)

    # L[i, j] will contain the length of the longest common subsequence of
    # x[0..i - 1] and y[0..j - 1].
    L = _np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                continue
            elif x[i - 1] == y[j - 1]:
                L[i, j] = L[i - 1, j - 1] + 1
            else:
                L[i, j] = max(L[i - 1, j], L[i, j - 1])

    ret = []

    i, j = m, n
    while i > 0 and j > 0:
        # If current character in x and y are same, then current character is
        # part of the longest common subsequence.
        if x[i - 1] == y[j - 1]:
            ret.append(x[i - 1])
            i, j = i - 1, j - 1
        # If not same, then find the larger of two and go in the direction of
        # larger value.
        elif L[i - 1, j] > L[i, j - 1]:
            i -= 1
        else:
            j -= 1

    return ret[::-1]