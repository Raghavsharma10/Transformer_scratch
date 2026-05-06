def LCP(SA):
    """
    Compute the longest common prefix for every adjacent suffixes.
    The result is a list of same size as SA.
    Given two suffixes at positions i and i+1,
    their LCP is stored at position i+1.
    A zero is stored at position 0 of the output.

    >>> SA=SuffixArray("abba", unit=UNIT_BYTE)
    >>> SA._LCP_values
    array('i', [0, 1, 0, 1])

    >>> SA=SuffixArray("", unit=UNIT_BYTE)
    >>> SA._LCP_values
    array('i')

    >>> SA=SuffixArray("", unit=UNIT_CHARACTER)
    >>> SA._LCP_values
    array('i')

    >>> SA=SuffixArray("", unit=UNIT_WORD)
    >>> SA._LCP_values
    array('i')

    >>> SA=SuffixArray("abab", unit=UNIT_BYTE)
    >>> SA._LCP_values
    array('i', [0, 2, 0, 1])
    """
    string = SA.string
    length = SA.length
    lcps = _array("i", [0] * length)
    SA = SA.SA

    if _trace:
        delta = max(length // 100, 1)
        for i, pos in enumerate(SA):
            if i % delta == 0:
                percent = float((i + 1) * 100) / length
                print >> _stderr, "Compute_LCP %.2f%% (%i/%i)\r" % (percent, i + 1, length),
            lcps[i] = _longestCommonPrefix(string, string, SA[i - 1], pos)
    else:
        for i, pos in enumerate(SA):
            lcps[i] = _longestCommonPrefix(string, string, SA[i - 1], pos)

    if _trace:
        print >> _stderr, "Compute_LCP %.2f%% (%i/%i)\r" % (100.0, length, length)

    if lcps:  # Correct the case where string[0] == string[-1]
        lcps[0] = 0
    return lcps