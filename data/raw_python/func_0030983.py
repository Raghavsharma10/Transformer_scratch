def matchToString(aaMatch, read1, read2, indent='', offsets=None):
    """
    Format amino acid sequence match as a string.

    @param aaMatch: A C{dict} returned by C{compareAaReads}.
    @param read1: A C{Read} instance or an instance of one of its subclasses.
    @param read2: A C{Read} instance or an instance of one of its subclasses.
    @param indent: A C{str} to indent all returned lines with.
    @param offsets: If not C{None}, a C{set} of offsets of interest that were
        only considered when making C{match}.
    @return: A C{str} describing the match.
    """
    match = aaMatch['match']
    matchCount = match['matchCount']
    gapMismatchCount = match['gapMismatchCount']
    gapGapMismatchCount = match['gapGapMismatchCount']
    nonGapMismatchCount = match['nonGapMismatchCount']

    if offsets:
        len1 = len2 = len(offsets)
    else:
        len1, len2 = map(len, (read1, read2))

    result = []
    append = result.append

    append(countPrint('%sMatches' % indent, matchCount, len1, len2))
    mismatchCount = (gapMismatchCount + gapGapMismatchCount +
                     nonGapMismatchCount)
    append(countPrint('%sMismatches' % indent, mismatchCount, len1, len2))
    append(countPrint('%s  Not involving gaps (i.e., conflicts)' % (indent),
                      nonGapMismatchCount, len1, len2))
    append(countPrint('%s  Involving a gap in one sequence' % indent,
                      gapMismatchCount, len1, len2))
    append(countPrint('%s  Involving a gap in both sequences' % indent,
                      gapGapMismatchCount, len1, len2))

    for read, key in zip((read1, read2), ('read1', 'read2')):
        append('%s  Id: %s' % (indent, read.id))
        length = len(read)
        append('%s    Length: %d' % (indent, length))
        gapCount = len(aaMatch[key]['gapOffsets'])
        append(countPrint('%s    Gaps' % indent, gapCount, length))
        if gapCount:
            append(
                '%s    Gap locations (1-based): %s' %
                (indent,
                 ', '.join(map(lambda offset: str(offset + 1),
                               sorted(aaMatch[key]['gapOffsets'])))))
        extraCount = aaMatch[key]['extraCount']
        if extraCount:
            append(countPrint('%s    Extra nucleotides at end' % indent,
                              extraCount, length))

    return '\n'.join(result)