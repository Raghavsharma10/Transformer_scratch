def eValueToBitScore(eValue, dbSize, dbSequenceCount, queryLength,
                     lengthAdjustment):
    """
    Convert an e-value to a bit score.

    @param eValue: The C{float} e-value to convert.
    @param dbSize: The C{int} total size of the database (i.e., the sum of
        the lengths of all sequences in the BLAST database).
    @param dbSequenceCount: The C{int} number of sequences in the database.
    @param queryLength: The C{int} length of the query.
    @param lengthAdjustment: The C{int} length adjustment (BLAST XML output
        calls this the Statistics_hsp-len).
    @return: A C{float} bit score.
    """
    effectiveDbSize = (
        (dbSize - dbSequenceCount * lengthAdjustment) *
        (queryLength - lengthAdjustment)
    )
    return -1.0 * (log(eValue / effectiveDbSize) / _LOG2)