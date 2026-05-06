def _pathogenSamplePlot(self, pathogenName, sampleNames, ax):
        """
        Make an image of a graph giving pathogen read count (Y axis) versus
        sample id (X axis).

        @param pathogenName: A C{str} pathogen name.
        @param sampleNames: A sorted C{list} of sample names.
        @param ax: A matplotlib C{axes} instance.
        """
        readCounts = []
        for i, sampleName in enumerate(sampleNames):
            try:
                readCount = self.pathogenNames[pathogenName][sampleName][
                    'uniqueReadCount']
            except KeyError:
                readCount = 0
            readCounts.append(readCount)

        highlight = 'r'
        normal = 'gray'
        sdMultiple = 2.5
        minReadsForHighlighting = 10
        highlighted = []

        if len(readCounts) == 1:
            if readCounts[0] > minReadsForHighlighting:
                color = [highlight]
                highlighted.append(sampleNames[0])
            else:
                color = [normal]
        else:
            mean = np.mean(readCounts)
            sd = np.std(readCounts)
            color = []
            for readCount, sampleName in zip(readCounts, sampleNames):
                if (readCount > (sdMultiple * sd) + mean and
                        readCount >= minReadsForHighlighting):
                    color.append(highlight)
                    highlighted.append(sampleName)
                else:
                    color.append(normal)

        nSamples = len(sampleNames)
        x = np.arange(nSamples)
        yMin = np.zeros(nSamples)
        ax.set_xticks([])
        ax.set_xlim((-0.5, nSamples - 0.5))
        ax.vlines(x, yMin, readCounts, color=color)
        if highlighted:
            title = '%s\nIn red: %s' % (
                pathogenName, fill(', '.join(highlighted), 50))
        else:
            # Add a newline to keep the first line of each title at the
            # same place as those titles that have an "In red:" second
            # line.
            title = pathogenName + '\n'

        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)