def makeFrequencyPanel(allFreqs, patientName):
    """
    For a title, make a graph showing the frequencies.

    @param allFreqs: result from getCompleteFreqs
    @param patientName: A C{str}, title for the panel
    """
    titles = sorted(
        iter(allFreqs.keys()),
        key=lambda title: (allFreqs[title]['bitScoreMax'], title))

    origMaxY = 0
    cols = 6
    rows = len(allFreqs)
    figure, ax = plt.subplots(rows, cols, squeeze=False)
    substitutions = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
    colors = ['blue', 'black', 'red', 'yellow', 'green', 'orange']

    for i, title in enumerate(titles):
        for index in range(6):
            for subst in allFreqs[str(title)]:
                substitution = substitutions[index]
                print(i, index, title, 'substitution', substitutions[index])
                if substitution[0] == 'C':
                    pattern = 'cPattern'
                else:
                    pattern = 'tPattern'
                maxY = makeFrequencyGraph(allFreqs, title, substitution,
                                          pattern, color=colors[index],
                                          createFigure=False, showFigure=False,
                                          readsAx=ax[i][index])
                if maxY > origMaxY:
                    origMaxY = maxY

            # add title for individual plot.
            # if used for other viruses, this will have to be adapted.
            if index == 0:
                gi = title.split('|')[1]
                titles = title.split(' ')
                try:
                    typeIndex = titles.index('type')
                except ValueError:
                    typeNumber = 'gi: %s' % gi
                else:
                    typeNumber = titles[typeIndex + 1]

                ax[i][index].set_ylabel(('Type %s \n maxBitScore: %s' % (
                    typeNumber, allFreqs[title]['bitScoreMax'])), fontsize=10)
            # add xAxis tick labels
            if i == 0:
                ax[i][index].set_title(substitution, fontsize=13)
            if i == len(allFreqs) - 1 or i == (len(allFreqs) - 1) / 2:
                if index < 3:
                    pat = ['ACA', 'ACC', 'ACG', 'ACT', 'CCA', 'CCC', 'CCG',
                           'CCT', 'GCA', 'GCC', 'GCG', 'GCT', 'TCA', 'TCC',
                           'TCG', 'TCT']
                else:
                    pat = ['ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG',
                           'CTT', 'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC',
                           'TTG', 'TTT']
                ax[i][index].set_xticklabels(pat, rotation=45, fontsize=8)

    # make Y-axis equal
    for i, title in enumerate(allFreqs):
        for index in range(6):
            a = ax[i][index]
            a.set_ylim([0, origMaxY])
    # add title of whole panel
    figure.suptitle('Mutation Signatures in %s' % patientName, fontsize=20)
    figure.set_size_inches(5 * cols, 3 * rows, forward=True)
    figure.show()

    return allFreqs