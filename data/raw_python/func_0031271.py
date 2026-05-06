def alignmentPanel(titlesAlignments, sortOn='maxScore', idList=False,
                   equalizeXAxes=False, xRange='subject', logLinearXAxis=False,
                   rankScores=False, showFeatures=True,
                   logBase=DEFAULT_LOG_LINEAR_X_AXIS_BASE):
    """
    Produces a rectangular panel of graphs that each contain an alignment graph
    against a given sequence.

    @param titlesAlignments: A L{dark.titles.TitlesAlignments} instance.
    @param sortOn: The attribute to sort subplots on. Either "maxScore",
        "medianScore", "readCount", "length", or "title".
    @param idList: A dictionary. Keys are colors and values are lists of read
        ids that should be colored using that color.
    @param equalizeXAxes: If C{True}, adjust the X axis on each alignment plot
        to be the same.
    @param xRange: Set to either 'subject' or 'reads' to indicate the range of
        the X axis.
    @param logLinearXAxis: If C{True}, convert read offsets so that empty
        regions in the plots we're preparing will only be as wide as their
        logged actual values.
    @param logBase: The logarithm base to use if logLinearXAxis is C{True}.
    @param: rankScores: If C{True}, change the scores for the reads for each
        title to be their rank (worst to best).
    @param showFeatures: If C{True}, look online for features of the subject
        sequences.
    @raise ValueError: If C{outputDir} exists but is not a directory or if
        C{xRange} is not "subject" or "reads".
    """

    if xRange not in ('subject', 'reads'):
        raise ValueError('xRange must be either "subject" or "reads".')

    start = time()
    titles = titlesAlignments.sortTitles(sortOn)
    cols = 5
    rows = int(len(titles) / cols) + (0 if len(titles) % cols == 0 else 1)
    figure, ax = plt.subplots(rows, cols, squeeze=False)
    allGraphInfo = {}
    coords = dimensionalIterator((rows, cols))

    report('Plotting %d titles in %dx%d grid, sorted on %s' %
           (len(titles), rows, cols, sortOn))

    for i, title in enumerate(titles):
        titleAlignments = titlesAlignments[title]
        row, col = next(coords)
        report('%d: %s %s' % (i, title, NCBISequenceLinkURL(title, '')))

        # Add a small plot to the alignment panel.
        graphInfo = alignmentGraph(
            titlesAlignments, title, addQueryLines=True,
            showFeatures=showFeatures, rankScores=rankScores,
            logLinearXAxis=logLinearXAxis, logBase=logBase,
            colorQueryBases=False, createFigure=False, showFigure=False,
            readsAx=ax[row][col], quiet=True, idList=idList, xRange=xRange,
            showOrfs=False)

        allGraphInfo[title] = graphInfo
        readCount = titleAlignments.readCount()
        hspCount = titleAlignments.hspCount()

        # Make a short title for the small panel blue plot, ignoring any
        # leading NCBI gi / accession numbers.
        if title.startswith('gi|') and title.find(' ') > -1:
            shortTitle = title.split(' ', 1)[1][:40]
        else:
            shortTitle = title[:40]

        plotTitle = ('%d: %s\nLength %d, %d read%s, %d HSP%s.' % (
            i, shortTitle, titleAlignments.subjectLength,
            readCount, '' if readCount == 1 else 's',
            hspCount, '' if hspCount == 1 else 's'))

        if hspCount:
            if rankScores:
                plotTitle += '\nY axis is ranked score'
            else:
                plotTitle += '\nmax %.2f, median %.2f' % (
                    titleAlignments.bestHsp().score.score,
                    titleAlignments.medianScore())

        ax[row][col].set_title(plotTitle, fontsize=10)

    maxX = max(graphInfo['maxX'] for graphInfo in allGraphInfo.values())
    minX = min(graphInfo['minX'] for graphInfo in allGraphInfo.values())
    maxY = max(graphInfo['maxY'] for graphInfo in allGraphInfo.values())
    minY = min(graphInfo['minY'] for graphInfo in allGraphInfo.values())

    # Post-process graphs to adjust axes, etc.

    coords = dimensionalIterator((rows, cols))
    for title in titles:
        titleAlignments = titlesAlignments[title]
        row, col = next(coords)
        a = ax[row][col]
        a.set_ylim([0, int(maxY * Y_AXIS_UPPER_PADDING)])
        if equalizeXAxes:
            a.set_xlim([minX, maxX])
        a.set_yticks([])
        a.set_xticks([])

        if xRange == 'subject' and minX < 0:
            # Add a vertical line at x=0 so we can see the 'whiskers' of
            # reads that extend to the left of the sequence we're aligning
            # against.
            a.axvline(x=0, color='#cccccc')

        # Add a line on the right of each sub-plot so we can see where the
        # sequence ends (as all panel graphs have the same width and we
        # otherwise couldn't tell).
        sequenceLen = titleAlignments.subjectLength
        if logLinearXAxis:
            sequenceLen = allGraphInfo[title]['adjustOffset'](sequenceLen)
        a.axvline(x=sequenceLen, color='#cccccc')

    # Hide the final panel graphs (if any) that have no content. We do this
    # because the panel is a rectangular grid and some of the plots at the
    # end of the last row may be unused.
    for row, col in coords:
        ax[row][col].axis('off')

    # plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.93,
    # wspace=0.1, hspace=None)
    plt.subplots_adjust(hspace=0.4)
    figure.suptitle('X: %d to %d, Y (%s): %d to %d' %
                    (minX, maxX,
                     titlesAlignments.readsAlignments.params.scoreTitle,
                     int(minY), int(maxY)), fontsize=20)
    figure.set_size_inches(5 * cols, 3 * rows, forward=True)
    figure.show()
    stop = time()
    report('Alignment panel generated in %.3f mins.' % ((stop - start) / 60.0))