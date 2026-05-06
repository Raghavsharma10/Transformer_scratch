def allExpressionLevels(self):
        """
        Return an iterator over all expression levels
        """
        for dataset in self.getDatasets():
            for rnaQuantificationSet in dataset.getRnaQuantificationSets():
                for rnaQuantification in \
                        rnaQuantificationSet.getRnaQuantifications():
                    for expressionLevel in \
                            rnaQuantification.getExpressionLevels():
                        yield expressionLevel