def allCallSets(self):
        """
        Return an iterator over all call sets in the data repo
        """
        for dataset in self.getDatasets():
            for variantSet in dataset.getVariantSets():
                for callSet in variantSet.getCallSets():
                    yield callSet