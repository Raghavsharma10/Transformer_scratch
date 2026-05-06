def makeFrequencyGraph(allFreqs, title, substitution, pattern,
                       color='blue', createFigure=True, showFigure=True,
                       readsAx=False):
    """
    For a title, make a graph showing the frequencies.

    @param allFreqs: result from getCompleteFreqs
    @param title: A C{str}, title of virus of which frequencies should be
        plotted.
    @param substitution: A C{str}, which substitution should be plotted;
        must be one of 'C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G'.
    @param pattern: A C{str}, which pattern we're looking for ( must be
        one of 'cPattern', 'tPattern')
    @param color: A C{str}, color of bars.
    @param createFigure: If C{True}, create a figure.
    @param showFigure: If C{True}, show the created figure.
    @param readsAx: If not None, use this as the subplot for displaying reads.
    """
    cPattern = ['ACA', 'ACC', 'ACG', 'ACT', 'CCA', 'CCC', 'CCG', 'CCT',
                'GCA', 'GCC', 'GCG', 'GCT', 'TCA', 'TCC', 'TCG', 'TCT']
    tPattern = ['ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG', 'CTT',
                'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC', 'TTG', 'TTT']

    # choose the right pattern
    if pattern == 'cPattern':
        patterns = cPattern
    else:
        patterns = tPattern

    fig = plt.figure(figsize=(10, 10))
    ax = readsAx or fig.add_subplot(111)
    # how many bars
    N = 16
    ind = np.arange(N)
    width = 0.4
    # make a list in the right order, so that it can be plotted easily
    divisor = allFreqs[title]['numberOfReads']
    toPlot = allFreqs[title][substitution]
    index = 0
    data = []
    for item in patterns:
        newData = toPlot[patterns[index]] / divisor
        data.append(newData)
        index += 1
    # create the bars
    ax.bar(ind, data, width, color=color)
    maxY = np.max(data) + 5
    # axes and labels
    if createFigure:
        title = title.split('|')[4][:50]
        ax.set_title('%s \n %s' % (title, substitution), fontsize=20)
        ax.set_ylim(0, maxY)
        ax.set_ylabel('Absolute Number of Mutations', fontsize=16)
        ax.set_xticks(ind + width)
        ax.set_xticklabels(patterns, rotation=45, fontsize=8)
    if createFigure is False:
        ax.set_xticks(ind + width)
        ax.set_xticklabels(patterns, rotation=45, fontsize=0)
    else:
        if showFigure:
            plt.show()
    return maxY