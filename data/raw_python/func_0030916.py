def countGaps(btopString):
    """
    Count the query and subject gaps in a BTOP string.

    @param btopString: A C{str} BTOP sequence.
    @raise ValueError: If L{parseBtop} finds an error in the BTOP string
        C{btopString}.
    @return: A 2-tuple of C{int}s, with the (query, subject) gaps counts as
        found in C{btopString}.
    """
    queryGaps = subjectGaps = 0
    for countOrMismatch in parseBtop(btopString):
        if isinstance(countOrMismatch, tuple):
            queryChar, subjectChar = countOrMismatch
            queryGaps += int(queryChar == '-')
            subjectGaps += int(subjectChar == '-')

    return (queryGaps, subjectGaps)