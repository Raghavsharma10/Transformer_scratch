def add(self, fig, title, minX, maxX, offsetAdjuster=None,
            sequenceFetcher=None):
        """
        Find the features for a sequence title. If there aren't too many, add
        the features to C{fig}. Return information about the features, as
        described below.

        @param fig: A matplotlib figure.
        @param title: A C{str} sequence title from a BLAST hit. Of the form
            'gi|63148399|gb|DQ011818.1| Description...'.
        @param minX: The smallest x coordinate.
        @param maxX: The largest x coordinate.
         @param offsetAdjuster: a function for adjusting feature X axis offsets
            for plotting.
        @param sequenceFetcher: A function that takes a sequence title and a
            database name and returns a C{Bio.SeqIO} instance. If C{None}, use
            L{dark.entrez.getSequence}.
        @return: If we seem to be offline, return C{None}. Otherwise, return
            a L{FeatureList} instance.
        """

        offsetAdjuster = offsetAdjuster or (lambda x: x)

        fig.set_title('Target sequence features', fontsize=self.TITLE_FONTSIZE)
        fig.set_yticks([])

        features = FeatureList(title, self.DATABASE, self.WANTED_TYPES,
                               sequenceFetcher=sequenceFetcher)

        if features.offline:
            fig.text(minX + (maxX - minX) / 3.0, 0,
                     'You (or Genbank) appear to be offline.',
                     fontsize=self.FONTSIZE)
            fig.axis([minX, maxX, -1, 1])
            return None

        # If no interesting features were found, display a message saying
        # so in the figure.  Otherwise, if we don't have too many features
        # to plot, add the feature info to the figure.
        nFeatures = len(features)
        if nFeatures == 0:
            # fig.text(minX + (maxX - minX) / 3.0, 0, 'No features found',
            #          fontsize=self.FONTSIZE)
            fig.text(0.5, 0.5, 'No features found',
                     horizontalalignment='center', verticalalignment='center',
                     transform=fig.transAxes, fontsize=self.FONTSIZE)
            fig.axis([minX, maxX, -1, 1])
        elif nFeatures <= self.MAX_FEATURES_TO_DISPLAY:
            # Call the method in our subclass to do the figure display.
            self._displayFeatures(fig, features, minX, maxX, offsetAdjuster)
        else:
            self.tooManyFeaturesToPlot = True
            # fig.text(minX + (maxX - minX) / 3.0, 0,
            # 'Too many features to plot.', fontsize=self.FONTSIZE)
            fig.text(0.5, 0.5, 'Too many features to plot',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=self.FONTSIZE, transform=fig.transAxes)
            fig.axis([minX, maxX, -1, 1])

        return features