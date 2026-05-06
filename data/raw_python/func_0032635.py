def labelAxes(self, numLines=(5,5)):
        """Put labels on axes

        Note: I should do better than this by picking round numbers
        as the places to put the labels.

        Note: If I ever do rotated projections, this simple approach
        will fail.
        """

        x1, x2, y1, y2 = mp.axis()
        ra1, dec0 = self.pixToSky(x1, y1)
        raRange, decRange = self.getRaDecRanges(numLines)
        ax = mp.gca()

        x_ticks = self.skyToPix(raRange, dec0)[0]
        y_ticks = self.skyToPix(ra1, decRange)[1]

        ax.xaxis.set_ticks(x_ticks)
        ax.xaxis.set_ticklabels([str(int(i)) for i in raRange])
        mp.xlabel("Right Ascension (deg)")

        ax.yaxis.set_ticks(y_ticks)
        ax.yaxis.set_ticklabels([str(int(i)) for i in decRange])
        mp.ylabel("Declination (deg)")