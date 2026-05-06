def collectData(reads1, reads2, square, matchAmbiguous):
    """
    Get pairwise matching statistics for two sets of reads.

    @param reads1: An C{OrderedDict} of C{str} read ids whose values are
        C{Read} instances. These will be the rows of the table.
    @param reads2: An C{OrderedDict} of C{str} read ids whose values are
        C{Read} instances. These will be the columns of the table.
    @param square: If C{True} we are making a square table of a set of
        sequences against themselves (in which case we show nothing on the
        diagonal).
    @param matchAmbiguous: If C{True}, count ambiguous nucleotides that are
        possibly correct as actually being correct. Otherwise, we are strict
        and insist that only non-ambiguous nucleotides can contribute to the
        matching nucleotide count.
    """
    result = defaultdict(dict)
    for id1, read1 in reads1.items():
        for id2, read2 in reads2.items():
            if id1 != id2 or not square:
                match = compareDNAReads(
                    read1, read2, matchAmbiguous=matchAmbiguous)['match']
                if not matchAmbiguous:
                    assert match['ambiguousMatchCount'] == 0
                result[id1][id2] = result[id2][id1] = match

    return result