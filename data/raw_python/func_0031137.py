def getAPOBECFrequencies(dotAlignment, orig, new, pattern):
    """
    Gets mutation frequencies if they are in a certain pattern.

    @param dotAlignment: result from calling basePlotter
    @param orig: A C{str}, naming the original base
    @param new: A C{str}, what orig was mutated to
    @param pattern: A C{str}m which pattern we're looking for
        (must be one of 'cPattern', 'tPattern')
    """
    cPattern = ['ACA', 'ACC', 'ACG', 'ACT', 'CCA', 'CCC', 'CCG', 'CCT',
                'GCA', 'GCC', 'GCG', 'GCT', 'TCA', 'TCC', 'TCG', 'TCT']
    tPattern = ['ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG', 'CTT',
                'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC', 'TTG', 'TTT']
    # choose the right pattern
    if pattern == 'cPattern':
        patterns = cPattern
        middleBase = 'C'
    else:
        patterns = tPattern
        middleBase = 'T'
    # generate the freqs dict with the right pattern
    freqs = defaultdict(int)
    for pattern in patterns:
        freqs[pattern] = 0
    # get the subject sequence from dotAlignment
    subject = dotAlignment[0].split('\t')[3]
    # exclude the subject from the dotAlignment, so just the queries
    # are left over
    queries = dotAlignment[1:]
    for item in queries:
        query = item.split('\t')[1]
        index = 0
        for queryBase in query:
            qBase = query[index]
            sBase = subject[index]
            if qBase == new and sBase == orig:
                try:
                    plusSb = subject[index + 1]
                    minusSb = subject[index - 1]
                except IndexError:
                    plusSb = 'end'
                motif = '%s%s%s' % (minusSb, middleBase, plusSb)
                if motif in freqs:
                    freqs[motif] += 1
            index += 1

    return freqs