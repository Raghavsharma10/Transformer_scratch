def get_daterange(self, dataset):
        """The method is getting information about date range of dataset"""

        path = '/api/1.0/meta/dataset/{}/daterange'
        return self._api_get(definition.DateRange, path.format(dataset))