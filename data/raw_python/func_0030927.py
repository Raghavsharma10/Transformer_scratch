def matchToString(dnaMatch, read1, read2, matchAmbiguous=True, indent='',
                  offsets=None):
    """
    Format a DNA match as a string.

    @param dnaMatch: A C{dict} returned by C{compareDNAReads}.
    @param read1: A C{Read} instance or an instance of one of its subclasses.
    @param read2: A C{Read} instance or an instance of one of its subclasses.
    @param matchAmbiguous: If C{True}, ambiguous nucleotides that are
        possibly correct were counted as actually being correct. Otherwise,
        the match was done strictly, insisting that only non-ambiguous
        nucleotides could contribute to the matching nucleotide count.
    @param indent: A C{str} to indent all returned lines with.
    @param offsets: If not C{None}, a C{set} of offsets of interest that were
        only considered when making C{match}.
    @return: A C{str} describing the match.
    """
    match = dnaMatch['match']
    identicalMatchCount = match['identicalMatchCount']
    ambiguousMatchCount = match['ambiguousMatchCount']
    gapMismatchCount = match['gapMismatchCount']
    gapGapMismatchCount = match['gapGapMismatchCount']
    nonGapMismatchCount = match['nonGapMismatchCount']

    if offsets:
        len1 = len2 = len(offsets)
    else:
        len1, len2 = map(len, (read1, read2))

    result = []
    append = result.append

    append(countPrint('%sExact matches' % indent, identicalMatchCount,
                      len1, len2))
    append(countPrint('%sAmbiguous matches' % indent, ambiguousMatchCount,
                      len1, len2))
    if ambiguousMatchCount and identicalMatchCount:
        anyMatchCount = identicalMatchCount + ambiguousMatchCount
        append(countPrint('%sExact or ambiguous matches' % indent,
                          anyMatchCount, len1, len2))

    mismatchCount = (gapMismatchCount + gapGapMismatchCount +
                     nonGapMismatchCount)
    append(countPrint('%sMismatches' % indent, mismatchCount, len1, len2))
    conflicts = 'conflicts' if matchAmbiguous else 'conflicts or ambiguities'
    append(countPrint('%s  Not involving gaps (i.e., %s)' % (indent,
                      conflicts), nonGapMismatchCount, len1, len2))
    append(countPrint('%s  Involving a gap in one sequence' % indent,
                      gapMismatchCount, len1, len2))
    append(countPrint('%s  Involving a gap in both sequences' % indent,
                      gapGapMismatchCount, len1, len2))

    for read, key in zip((read1, read2), ('read1', 'read2')):
        append('%s  Id: %s' % (indent, read.id))
        length = len(read)
        append('%s    Length: %d' % (indent, length))
        gapCount = len(dnaMatch[key]['gapOffsets'])
        append(countPrint('%s    Gaps' % indent, gapCount, length))
        if gapCount:
            append(
                '%s    Gap locations (1-based): %s' %
                (indent,
                 ', '.join(map(lambda offset: str(offset + 1),
                               sorted(dnaMatch[key]['gapOffsets'])))))
        ambiguousCount = len(dnaMatch[key]['ambiguousOffsets'])
        append(countPrint('%s    Ambiguous' % indent, ambiguousCount, length))
        extraCount = dnaMatch[key]['extraCount']
        if extraCount:
            append(countPrint('%s    Extra nucleotides at end' % indent,
                              extraCount, length))

    return '\n'.join(result)