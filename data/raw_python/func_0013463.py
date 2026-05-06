def continuousSetsGenerator(self, request):
        """
        Returns a generator over the (continuousSet, nextPageToken) pairs
        defined by the specified request.
        """
        dataset = self.getDataRepository().getDataset(request.dataset_id)
        return self._topLevelObjectGenerator(
            request, dataset.getNumContinuousSets(),
            dataset.getContinuousSetByIndex)