def getCompleteFreqs(blastHits):
    """
    Make a dictionary which collects all mutation frequencies from
    all reads.
    Calls basePlotter to get dotAlignment, which is passed to
    getAPOBECFrequencies with the respective parameter, to collect
    the frequencies.

    @param blastHits: A L{dark.blast.BlastHits} instance.
    """
    allFreqs = {}
    for title in blastHits.titles:
        allFreqs[title] = {
            'C>A': {},
            'C>G': {},
            'C>T': {},
            'T>A': {},
            'T>C': {},
            'T>G': {},
        }
        basesPlotted = basePlotter(blastHits, title)
        for mutation in allFreqs[title]:
            orig = mutation[0]
            new = mutation[2]
            if orig == 'C':
                pattern = 'cPattern'
            else:
                pattern = 'tPattern'
            freqs = getAPOBECFrequencies(basesPlotted, orig, new, pattern)
            allFreqs[title][mutation] = freqs
        numberOfReads = len(blastHits.titles[title]['plotInfo']['items'])
        allFreqs[title]['numberOfReads'] = numberOfReads
        allFreqs[title]['bitScoreMax'] = blastHits.titles[
            title]['plotInfo']['bitScoreMax']
    return allFreqs