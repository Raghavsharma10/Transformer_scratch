def htmlTable(tableData, reads1, reads2, square, matchAmbiguous, colors,
              concise=False, showLengths=False, showGaps=False, showNs=False,
              footer=False, div=False, gapChars='-'):
    """
    Make an HTML table showing inter-sequence distances.

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
    @param colors: A C{list} of (threshold, color) tuples, where threshold is a
        C{float} and color is a C{str} to be used as a cell background. This
        is as returned by C{parseColors}.
    @param concise: If C{True}, do not show match details.
    @param showLengths: If C{True}, include the lengths of sequences.
    @param showGaps: If C{True}, include the number of gaps in sequences.
    @param showGaps: If C{True}, include the number of N characters in
        sequences.
    @param footer: If C{True}, incude a footer row giving the same information
        as found in the table header.
    @param div: If C{True}, return an HTML <div> fragment only, not a full HTML
        document.
    @param gapChars: A C{str} of sequence characters considered to be gaps.
    @return: An HTML C{str} showing inter-sequence distances.
    """
    readLengths1 = getReadLengths(reads1.values(), gapChars)
    readLengths2 = getReadLengths(reads2.values(), gapChars)
    result = []
    append = result.append

    def writeHeader():
        # The header row of the table.
        append('    <tr>')
        append('    <td>&nbsp;</td>')
        for read2 in reads2.values():
            append('    <td class="title"><span class="name">%s</span>' %
                   read2.id)
            if showLengths and not square:
                append('    <br>L:%d' % readLengths2[read2.id])
            if showGaps and not square:
                append('    <br>G:%d' % (len(read2) - readLengths2[read2.id]))
            if showNs and not square:
                append('    <br>N:%d' % read2.sequence.count('N'))
            append('    </td>')
        append('    </tr>')

    if div:
        append('<div>')
    else:
        append('<!DOCTYPE HTML>')
        append('<html>')
        append('<head>')
        append('<meta charset="UTF-8">')
        append('</head>')
        append('<body>')

    append('<style>')
    append("""
        table {
            border-collapse: collapse;
        }
        table, td {
            border: 1px solid #ccc;
        }
        tr:hover {
            background-color: #f2f2f2;
        }
        td {
            vertical-align: top;
            font-size: 14px;
        }
        span.name {
            font-weight: bold;
        }
        span.best {
            font-weight: bold;
        }
    """)

    # Add color style information for the identity thresholds.
    for threshold, color in colors:
        append('.%s { background-color: %s; }' % (
            thresholdToCssName(threshold), color))

    append('</style>')

    if not div:
        append(explanation(
            matchAmbiguous, concise, showLengths, showGaps, showNs))
    append('<div style="overflow-x:auto;">')
    append('<table>')
    append('  <tbody>')

    # Pre-process to find the best identities in each sample row.
    bestIdentityForId = {}

    for id1, read1 in reads1.items():
        # Look for best identity for the sample.
        read1Len = readLengths1[id1]
        bestIdentity = -1.0
        for id2, read2 in reads2.items():
            if id1 != id2 or not square:
                stats = tableData[id1][id2]
                identity = (
                    stats['identicalMatchCount'] +
                    (stats['ambiguousMatchCount'] if matchAmbiguous else 0)
                ) / read1Len
                if identity > bestIdentity:
                    bestIdentity = identity
        bestIdentityForId[id1] = bestIdentity

    writeHeader()

    # The main body of the table.
    for id1, read1 in reads1.items():
        read1Len = readLengths1[id1]
        append('    <tr>')
        append('      <td class="title"><span class="name">%s</span>' % id1)
        if showLengths:
            append('<br/>L:%d' % read1Len)
        if showGaps:
            append('<br/>G:%d' % (len(read1) - read1Len))
        if showNs:
            append('<br/>N:%d' % read1.sequence.count('N'))
        append('</td>')
        for id2, read2 in reads2.items():
            if id1 == id2 and square:
                append('<td>&nbsp;</td>')
                continue

            stats = tableData[id1][id2]
            identity = (
                stats['identicalMatchCount'] +
                (stats['ambiguousMatchCount'] if matchAmbiguous else 0)
            ) / read1Len

            append('      <td class="%s">' % thresholdToCssName(
                thresholdForIdentity(identity, colors)))

            # The maximum percent identity.
            if identity == bestIdentityForId[id1]:
                scoreStyle = ' class="best"'
            else:
                scoreStyle = ''

            append('<span%s>%.4f</span>' % (scoreStyle, identity))

            if not concise:
                append('<br/>IM:%d' % stats['identicalMatchCount'])

                if matchAmbiguous:
                    append('<br/>AM:%d' % stats['ambiguousMatchCount'])

                append(
                    '<br/>GG:%d'
                    '<br/>G?:%d'
                    '<br/>NE:%d' %
                    (stats['gapGapMismatchCount'],
                     stats['gapMismatchCount'],
                     stats['nonGapMismatchCount']))
            append('      </td>')
        append('    </tr>')

    if footer:
        writeHeader()

    append('  </tbody>')
    append('</table>')
    append('</div>')

    if div:
        append('</div>')
    else:
        append('</body>')
        append('</html>')

    return '\n'.join(result)