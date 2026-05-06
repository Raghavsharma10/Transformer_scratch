def rnaQuantificationSetsGenerator(self, request):
        """
        Returns a generator over the (rnaQuantificationSet, nextPageToken)
        pairs defined by the specified request.
        """
        dataset = self.getDataRepository().getDataset(request.dataset_id)
        return self._topLevelObjectGenerator(
            request, dataset.getNumRnaQuantificationSets(),
            dataset.getRnaQuantificationSetByIndex)