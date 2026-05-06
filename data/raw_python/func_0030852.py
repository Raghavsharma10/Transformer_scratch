def adjustPlot(self, readsAx):
        """
        Add a horizontal line to the plotted reads if we're plotting e-values
        and a zero e-value was found.

        @param readsAx: A Matplotlib sub-plot instance, as returned by
            matplotlib.pyplot.subplot.
        """
        # If we're using bit scores, there's nothing to do.
        if self.scoreClass is HigherIsBetterScore:
            return

        if self._zeroEValueFound:
            readsAx.axhline(y=self._maxConvertedEValue + 0.5, color='#cccccc',
                            linewidth=0.5)