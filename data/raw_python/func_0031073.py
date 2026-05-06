def pathogenPanel(self, filename):
        """
        Make a panel of images, with each image being a graph giving pathogen
        de-duplicated (by id) read count (Y axis) versus sample id (X axis).

        @param filename: A C{str} file name to write the image to.
        """
        import matplotlib
        matplotlib.use('PDF')
        import matplotlib.pyplot as plt

        self._computeUniqueReadCounts()
        pathogenNames = sorted(self.pathogenNames)
        sampleNames = sorted(self.sampleNames)

        cols = 5
        rows = int(len(pathogenNames) / cols) + (
            0 if len(pathogenNames) % cols == 0 else 1)
        figure, ax = plt.subplots(rows, cols, squeeze=False)

        coords = dimensionalIterator((rows, cols))

        for i, pathogenName in enumerate(pathogenNames):
            row, col = next(coords)
            self._pathogenSamplePlot(pathogenName, sampleNames, ax[row][col])

        # Hide the final panel graphs (if any) that have no content. We do
        # this because the panel is a rectangular grid and some of the
        # plots at the end of the last row may be unused.
        for row, col in coords:
            ax[row][col].axis('off')

        figure.suptitle(
            ('Per-sample read count for %d pathogen%s and %d sample%s.\n\n'
             'Sample name%s: %s') % (
                 len(pathogenNames),
                 '' if len(pathogenNames) == 1 else 's',
                 len(sampleNames),
                 '' if len(sampleNames) == 1 else 's',
                 '' if len(sampleNames) == 1 else 's',
                 fill(', '.join(sampleNames), 50)),
            fontsize=20)
        figure.set_size_inches(5.0 * cols, 2.0 * rows, forward=True)
        plt.subplots_adjust(hspace=0.4)

        figure.savefig(filename)