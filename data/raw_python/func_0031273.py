def scoreGraph(titlesAlignments, find=None, showTitles=False, figureWidth=5,
               figureHeight=5):
    """
    NOTE: This function has probably bit rotted (but only a little).

    Produce a rectangular panel of graphs, each of which shows sorted scores
    for a title. Matches against a certain sequence title, as determined by
    C{find}, (see below) are highlighted.

    @param find: A function that can be passed a sequence title. If the
        function returns C{True} a red dot is put into the graph at that point
        to highlight the match.
    @param showTitles: If C{True} display read sequence names. The panel tends
        to look terrible when titles are displayed. If C{False}, show no title.
    @param figureWidth: The C{float} width of the figure, in inches.
    @param figureHeight: The C{float} height of the figure, in inches.
    """
    maxScore = None
    maxHsps = 0
    cols = 5
    rows = int(len(titlesAlignments) / cols) + (
        0 if len(titlesAlignments) % cols == 0 else 1)
    f, ax = plt.subplots(rows, cols)
    coords = dimensionalIterator((rows, cols))

    for title in titlesAlignments:
        titleAlignments = titlesAlignments[title]
        row, col = next(coords)
        hspCount = titleAlignments.hspCount()
        if hspCount > maxHsps:
            maxHsps = hspCount
        scores = []
        highlightX = []
        highlightY = []
        for x, titleAlignment in enumerate(titleAlignments):
            score = titleAlignment.hsps[0].score.score
            scores.append(score)
            if find and find(titleAlignment.subjectTitle):
                highlightX.append(x)
                highlightY.append(score)
        a = ax[row][col]
        if scores:
            max_ = max(scores)
            if maxScore is None or max_ > maxScore:
                maxScore = max_
            x = np.arange(0, len(scores))
            a.plot(x, scores)
        if highlightX:
            a.plot(highlightX, highlightY, 'ro')
        if showTitles:
            a.set_title('%s' % title, fontsize=10)

    # Adjust all plots to have the same dimensions.
    coords = dimensionalIterator((rows, cols))
    for _ in range(len(titlesAlignments)):
        row, col = next(coords)
        a = ax[row][col]
        a.axis([0, maxHsps, 0, maxScore])
        # a.set_yscale('log')
        a.set_yticks([])
        a.set_xticks([])

    # Hide the final panel graphs (if any) that have no content. We do this
    # because the panel is a rectangular grid and some of the plots at the
    # end of the last row may be unused.
    for row, col in coords:
        ax[row][col].axis('off')

    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.93,
                        wspace=0.1, hspace=None)
    f.suptitle('max HSPs %d, max score %f' % (maxHsps, maxScore))
    f.set_size_inches(figureWidth, figureHeight, forward=True)
    # f.savefig('scores.png')
    plt.show()