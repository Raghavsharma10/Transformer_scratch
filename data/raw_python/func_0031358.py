def normalizeHSP(hsp, queryLen, diamondTask):
    """
    Examine an HSP and return information about where the query and subject
    match begins and ends.  Return a dict with keys that allow the query to
    be displayed against the subject. The returned readStartInSubject and
    readEndInSubject indices are offsets into the subject. I.e., they
    indicate where in the subject the query falls.

    In the returned object, all indices are suitable for Python string
    slicing etc.  We must be careful to convert from the 1-based offsets
    found in DIAMOND output properly.

    hsp['frame'] is a value from {-3, -2, -1, 1, 2, 3}. The sign indicates
    negative or positive sense (i.e., the direction of reading through the
    query to get the alignment). The frame value is the nucleotide match offset
    modulo 3, plus one (i.e., it tells us which of the 3 possible query reading
    frames was used in the match).

    NOTE: the returned readStartInSubject value may be negative.  We consider
    the subject sequence to start at offset 0.  So if the query string has
    sufficient additional nucleotides before the start of the alignment
    match, it may protrude to the left of the subject. Similarly, the returned
    readEndInSubject can be greater than the subjectEnd.

    @param hsp: an HSP in the form of a C{dict}, built from a DIAMOND record.
        All passed offsets are 1-based.
    @param queryLen: the length of the query sequence.
    @param diamondTask: The C{str} command-line matching algorithm that was
        run (either 'blastx' or 'blastp').
    @return: A C{dict} with C{str} keys and C{int} offset values. Keys are
            readStart
            readEnd
            readStartInSubject
            readEndInSubject
            subjectStart
            subjectEnd
        The returned offset values are all zero-based.
    """

    queryGaps, subjectGaps = countGaps(hsp['btop'])

    # Make some variables using Python's standard string indexing (start
    # offset included, end offset not). No calculations in this function
    # are done with the original 1-based HSP variables.
    queryStart = hsp['query_start'] - 1
    queryEnd = hsp['query_end']
    subjectStart = hsp['sbjct_start'] - 1
    subjectEnd = hsp['sbjct_end']

    queryReversed = hsp['frame'] < 0

    # Query offsets must be ascending, unless we're looking at blastx output
    # and the query was reversed for the match.
    if queryStart >= queryEnd:
        if diamondTask == 'blastx' and queryReversed:
            # Compute new query start and end indices, based on their
            # distance from the end of the string.
            #
            # Above we took one off the start index, so we need to undo
            # that (because the start is actually the end). We didn't take
            # one off the end index, and need to do that now (because the
            # end is actually the start).
            queryStart = queryLen - (queryStart + 1)
            queryEnd = queryLen - (queryEnd - 1)
        else:
            _debugPrint(hsp, queryLen, locals(), 'queryStart >= queryEnd')

    if diamondTask == 'blastx':
        # In DIAMOND blastx output, subject offsets are based on protein
        # sequence length but queries (and the reported offsets) are
        # nucleotide.  Convert the query offsets to protein because we will
        # plot against the subject (protein).
        #
        # Convert queryLen and the query nucleotide start and end offsets
        # to be valid for the query after translation to AAs. When
        # translating, DIAMOND may ignore some nucleotides at the start
        # and/or the end of the original DNA query. At the start this is
        # due to the frame in use, and at the end it is due to always using
        # three nucleotides at a time to form codons.
        #
        # So, for example, a query of 6 nucleotides that is translated in
        # frame 2 (i.e., the translation starts from the second nucleotide)
        # will have length 1 as an AA sequence. The first nucleotide is
        # ignored due to the frame and the last two due to there not being
        # enough final nucleotides to make another codon.
        #
        # In the following, the subtraction accounts for the first form of
        # loss and the integer division for the second.
        initiallyIgnored = abs(hsp['frame']) - 1
        queryLen = (queryLen - initiallyIgnored) // 3
        queryStart = (queryStart - initiallyIgnored) // 3
        queryEnd = (queryEnd - initiallyIgnored) // 3

    # unmatchedQueryLeft is the number of query bases that will extend
    # to the left of the start of the subject in our plots.
    unmatchedQueryLeft = queryStart

    # Set the query offsets into the subject.
    queryStartInSubject = subjectStart - unmatchedQueryLeft
    queryEndInSubject = queryStartInSubject + queryLen + queryGaps

    _sanityCheck(subjectStart, subjectEnd, queryStart, queryEnd,
                 queryStartInSubject, queryEndInSubject, hsp, queryLen,
                 subjectGaps, queryGaps, locals())

    return {
        'readStart': queryStart,
        'readEnd': queryEnd,
        'readStartInSubject': queryStartInSubject,
        'readEndInSubject': queryEndInSubject,
        'subjectStart': subjectStart,
        'subjectEnd': subjectEnd,
    }