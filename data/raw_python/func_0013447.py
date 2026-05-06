def phenotypeAssociationSetsGenerator(self, request):
        """
        Returns a generator over the (phenotypeAssociationSet, nextPageToken)
        pairs defined by the specified request
        """
        dataset = self.getDataRepository().getDataset(request.dataset_id)
        return self._topLevelObjectGenerator(
            request, dataset.getNumPhenotypeAssociationSets(),
            dataset.getPhenotypeAssociationSetByIndex)