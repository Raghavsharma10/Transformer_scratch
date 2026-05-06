def simpleTable(tableData, reads1, reads2, square, matchAmbiguous, gapChars):
    """
    Make a text table showing inter-sequence distances.

    @param tableData: A C{defaultdict(dict)} keyed by read ids, whose values
        are the dictionaries returned by compareDNAReads.
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
    @param gapChars: A C{str} of sequence characters considered to be gaps.
    """
    readLengths1 = getReadLengths(reads1.values(), gapChars)
    print('ID\t' + '\t'.join(reads2))

    for id1, read1 in reads1.items():
        read1Len = readLengths1[id1]
        print(id1, end='')
        for id2, read2 in reads2.items():
            if id1 == id2 and square:
                print('\t', end='')
            else:
                stats = tableData[id1][id2]
                identity = (
                    stats['identicalMatchCount'] +
                    (stats['ambiguousMatchCount'] if matchAmbiguous else 0)
                ) / read1Len
                print('\t%.4f' % identity, end='')
        print()