def getExpressionLevels(
            self, threshold=0.0, names=[], startIndex=0, maxResults=0):
        """
        Returns the list of ExpressionLevels in this RNA Quantification.
        """
        rnaQuantificationId = self.getLocalId()
        with self._db as dataSource:
            expressionsReturned = dataSource.searchExpressionLevelsInDb(
                rnaQuantificationId,
                names=names,
                threshold=threshold,
                startIndex=startIndex,
                maxResults=maxResults)
            expressionLevels = [
                SqliteExpressionLevel(self, expressionEntry) for
                expressionEntry in expressionsReturned]
            return expressionLevels