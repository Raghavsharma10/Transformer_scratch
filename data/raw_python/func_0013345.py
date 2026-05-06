def allVariantAnnotationSets(self):
        """
        Return an iterator over all variant annotation sets
        in the data repo
        """
        for dataset in self.getDatasets():
            for variantSet in dataset.getVariantSets():
                for vaSet in variantSet.getVariantAnnotationSets():
                    yield vaSet