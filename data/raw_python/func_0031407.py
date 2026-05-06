def titleCounts(readsAlignments):
    """
    Count the number of times each title in a readsAlignments instance is
    matched. This is useful for rapidly discovering what titles were matched
    and with what frequency.

    @param readsAlignments: A L{dark.alignments.ReadsAlignments} instance.
    @return: A C{dict} whose keys are titles and whose values are the integer
        counts of the number of reads that matched that title.
    """

    titles = defaultdict(int)
    for readAlignments in readsAlignments:
        for alignment in readAlignments:
            titles[alignment.subjectTitle] += 1
    return titles