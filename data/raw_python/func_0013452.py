def variantSetsGenerator(self, request):
        """
        Returns a generator over the (variantSet, nextPageToken) pairs defined
        by the specified request.
        """
        dataset = self.getDataRepository().getDataset(request.dataset_id)
        return self._topLevelObjectGenerator(
            request, dataset.getNumVariantSets(),
            dataset.getVariantSetByIndex)