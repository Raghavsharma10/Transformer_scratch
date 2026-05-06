def getReadLengths(reads, gapChars):
    """
    Get all read lengths, excluding gap characters.

    @param reads: A C{Reads} instance.
    @param gapChars: A C{str} of sequence characters considered to be gaps.
    @return: A C{dict} keyed by read id, with C{int} length values.
    """
    gapChars = set(gapChars)
    result = {}
    for read in reads:
        result[read.id] = len(read) - sum(
            character in gapChars for character in read.sequence)
    return result