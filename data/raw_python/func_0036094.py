def _processResults(self):
        """ Checks each result can meet SNR requirments, adds to count
        :return:
        """

        resultsByClass = self._genEmptyResults()

        for astroObject in self.objectList:
            sortKey = self._getSortKey(astroObject)
            resultsByClass[sortKey] += 1

        return resultsByClass