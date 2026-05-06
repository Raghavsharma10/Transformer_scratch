def readGroupSetsGenerator(self, request):
        """
        Returns a generator over the (readGroupSet, nextPageToken) pairs
        defined by the specified request.
        """
        dataset = self.getDataRepository().getDataset(request.dataset_id)
        return self._readGroupSetsGenerator(
            request, dataset.getNumReadGroupSets(),
            dataset.getReadGroupSetByIndex)