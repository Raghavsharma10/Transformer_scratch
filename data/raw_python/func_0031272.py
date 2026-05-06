def alignmentPanelHTML(titlesAlignments, sortOn='maxScore',
                       outputDir=None, idList=False, equalizeXAxes=False,
                       xRange='subject', logLinearXAxis=False,
                       logBase=DEFAULT_LOG_LINEAR_X_AXIS_BASE,
                       rankScores=False, showFeatures=True, showOrfs=True):
    """
    Produces an HTML index file in C{outputDir} and a collection of alignment
    graphs and FASTA files to summarize the information in C{titlesAlignments}.

    @param titlesAlignments: A L{dark.titles.TitlesAlignments} instance.
    @param sortOn: The attribute to sort subplots on. Either "maxScore",
        "medianScore", "readCount", "length", or "title".
    @param outputDir: Specifies a C{str} directory to write the HTML to. If
        the directory does not exist it will be created.
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
    @param showOrfs: If C{True}, open reading frames will be displayed.
    @raise TypeError: If C{outputDir} is C{None}.
    @raise ValueError: If C{outputDir} is None or exists but is not a
        directory or if C{xRange} is not "subject" or "reads".
    """

    if xRange not in ('subject', 'reads'):
        raise ValueError('xRange must be either "subject" or "reads".')

    if equalizeXAxes:
        raise NotImplementedError('This feature is not yet implemented.')

    titles = titlesAlignments.sortTitles(sortOn)

    if os.access(outputDir, os.F_OK):
        # outputDir exists. Check it's a directory.
        if not S_ISDIR(os.stat(outputDir).st_mode):
            raise ValueError("%r is not a directory." % outputDir)
    else:
        if outputDir is None:
            raise ValueError("The outputDir needs to be specified.")
        else:
            os.mkdir(outputDir)

    htmlWriter = AlignmentPanelHTMLWriter(outputDir, titlesAlignments)

    for i, title in enumerate(titles):
        # titleAlignments = titlesAlignments[title]

        # If we are writing data to a file too, create a separate file with
        # a plot (this will be linked from the summary HTML).
        imageBasename = '%d.png' % i
        imageFile = '%s/%s' % (outputDir, imageBasename)
        graphInfo = alignmentGraph(
            titlesAlignments, title, addQueryLines=True,
            showFeatures=showFeatures, rankScores=rankScores,
            logLinearXAxis=logLinearXAxis, logBase=logBase,
            colorQueryBases=False, showFigure=False, imageFile=imageFile,
            quiet=True, idList=idList, xRange=xRange, showOrfs=showOrfs)

        # Close the image plot to make sure memory is flushed.
        plt.close()
        htmlWriter.addImage(imageBasename, title, graphInfo)

    htmlWriter.close()