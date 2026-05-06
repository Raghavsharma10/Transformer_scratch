def datasetsGenerator(self, request):
        """
        Returns a generator over the (dataset, nextPageToken) pairs
        defined by the specified request
        """
        return self._topLevelObjectGenerator(
            request, self.getDataRepository().getNumDatasets(),
            self.getDataRepository().getDatasetByIndex)