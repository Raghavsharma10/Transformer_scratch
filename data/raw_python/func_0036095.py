def _getPlotData(self):
        """ Turns the resultsByClass Dict into a list of bin groups skipping the uncertain group if empty

        return: (label list, ydata list)
        :rtype: tuple(list(str), list(float))
        """
        resultsByClass = self.resultsByClass

        try:
            if resultsByClass['Uncertain'] == 0:  # remove uncertain tag if present and = 0
                resultsByClass.pop('Uncertain', None)
        except KeyError:
            pass

        plotData = list(zip(*resultsByClass.items()))  # (labels, ydata)

        return plotData