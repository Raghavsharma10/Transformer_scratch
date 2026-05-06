def _sanityCheck(subjectStart, subjectEnd, queryStart, queryEnd,
                 queryStartInSubject, queryEndInSubject, hsp, queryLen,
                 subjectGaps, queryGaps, localDict):
    """
    Perform some sanity checks on an HSP. Call _debugPrint on any error.

    @param subjectStart: The 0-based C{int} start offset of the match in the
        subject.
    @param subjectEnd: The 0-based C{int} end offset of the match in the
        subject.
    @param queryStart: The 0-based C{int} start offset of the match in the
        query.
    @param queryEnd: The 0-based C{int} end offset of the match in the query.
    @param queryStartInSubject: The 0-based C{int} offset of where the query
        starts in the subject.
    @param queryEndInSubject: The 0-based C{int} offset of where the query
        ends in the subject.
    @param hsp: The HSP C{dict} passed to normalizeHSP.
    @param queryLen: the C{int} length of the query sequence.
    @param subjectGaps: the C{int} number of gaps in the subject.
    @param queryGaps: the C{int} number of gaps in the query.
    @param localDict: A C{dict} of local variables from our caller (as
        produced by locals()).
    """
    # Subject indices must always be ascending.
    if subjectStart >= subjectEnd:
        _debugPrint(hsp, queryLen, localDict, 'subjectStart >= subjectEnd')

    subjectMatchLength = subjectEnd - subjectStart
    queryMatchLength = queryEnd - queryStart

    # Sanity check that the length of the matches in the subject and query
    # are identical, taking into account gaps in both.
    subjectMatchLengthWithGaps = subjectMatchLength + subjectGaps
    queryMatchLengthWithGaps = queryMatchLength + queryGaps
    if subjectMatchLengthWithGaps != queryMatchLengthWithGaps:
        _debugPrint(hsp, queryLen, localDict,
                    'Including gaps, subject match length (%d) != Query match '
                    'length (%d)' % (subjectMatchLengthWithGaps,
                                     queryMatchLengthWithGaps))

    if queryStartInSubject > subjectStart:
        _debugPrint(hsp, queryLen, localDict,
                    'queryStartInSubject (%d) > subjectStart (%d)' %
                    (queryStartInSubject, subjectStart))
    if queryEndInSubject < subjectEnd:
        _debugPrint(hsp, queryLen, localDict,
                    'queryEndInSubject (%d) < subjectEnd (%d)' %
                    (queryEndInSubject, subjectEnd))