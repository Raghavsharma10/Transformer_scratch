def adjustHspsForPlotting(self, titleAlignments):
        """
        Our HSPs are about to be plotted. If we are using e-values, these need
        to be adjusted.

        @param titleAlignments: An instance of L{TitleAlignment}.
        """
        # If we're using bit scores, there's nothing to do.
        if self.scoreClass is HigherIsBetterScore:
            return

        # Convert all e-values to high positive values, and keep track of the
        # maximum converted value.
        maxConvertedEValue = None
        zeroHsps = []

        # Note: don't call self.hsps() here because that will read them
        # from disk again, which is not what's wanted.
        for hsp in titleAlignments.hsps():
            if hsp.score.score == 0.0:
                zeroHsps.append(hsp)
            else:
                convertedEValue = -1.0 * log10(hsp.score.score)
                hsp.score.score = convertedEValue
                if (maxConvertedEValue is None or
                        convertedEValue > maxConvertedEValue):
                    maxConvertedEValue = convertedEValue

        if zeroHsps:
            # Save values so that we can use them in self.adjustPlot
            self._maxConvertedEValue = maxConvertedEValue
            self._zeroEValueFound = True

            # Adjust all zero e-value HSPs to have numerically high values.
            if self.randomizeZeroEValues:
                for hsp in zeroHsps:
                    hsp.score.score = (maxConvertedEValue + 2 + uniform(
                        0, ZERO_EVALUE_UPPER_RANDOM_INCREMENT))
            else:
                for count, hsp in enumerate(zeroHsps, start=1):
                    hsp.score.score = maxConvertedEValue + count
        else:
            self._zeroEValueFound = False