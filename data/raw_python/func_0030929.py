def findKozakConsensus(read):
    """
    In a given DNA sequence, search for a Kozak consensus: (gcc)gccRccATGG.
    The upper case bases in that pattern are required, and the lower case
    bases are the ones most frequently found at the given positions. The
    initial 'gcc' sequence (in parentheses) is of uncertain significance
    and is not taken into account here.

    @param read: A C{DNARead} instance to be checked for Kozak consensi.
    @return: A generator that yields C{DNAKozakRead} instances.
    """
    readLen = len(read)
    if readLen > 9:
        offset = 6
        readSeq = read.sequence
        while offset < readLen - 3:
            triplet = readSeq[offset:offset + 3]
            if triplet == 'ATG':
                if readSeq[offset + 3] == 'G':
                    if readSeq[offset - 3] in 'GA':
                        kozakQualityCount = sum((
                            readSeq[offset - 1] == 'C',
                            readSeq[offset - 2] == 'C',
                            readSeq[offset - 4] == 'C',
                            readSeq[offset - 5] == 'C',
                            readSeq[offset - 6] == 'G'))

                        kozakQualityPercent = kozakQualityCount / 5.0 * 100
                        yield DNAKozakRead(read, offset - 6, offset + 4,
                                           kozakQualityPercent)
            offset += 1