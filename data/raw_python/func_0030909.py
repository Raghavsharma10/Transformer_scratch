def _sortHTML(titlesAlignments, by, limit=None):
    """
    Return an C{IPython.display.HTML} object with the alignments sorted by the
    given attribute.

    @param titlesAlignments: A L{dark.titles.TitlesAlignments} instance.
    @param by: A C{str}, one of 'length', 'maxScore', 'medianScore',
        'readCount', or 'title'.
    @param limit: An C{int} limit on the number of results to show.
    @return: An HTML instance with sorted titles and information about
        hit read count, length, and e-values.
    """
    out = []
    for i, title in enumerate(titlesAlignments.sortTitles(by), start=1):
        if limit is not None and i > limit:
            break
        titleAlignments = titlesAlignments[title]
        link = NCBISequenceLink(title, title)
        out.append(
            '%3d: reads=%d, len=%d, max=%s median=%s<br/>'
            '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%s' %
            (i, titleAlignments.readCount(), titleAlignments.subjectLength,
             titleAlignments.bestHsp().score.score,
             titleAlignments.medianScore(), link))
    return HTML('<pre>' + '<br/>'.join(out) + '</pre>')