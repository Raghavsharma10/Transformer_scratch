def get_dataset(self, datasetid):
        """The method is getting information about dataset byt it's id"""

        path = '/api/1.0/meta/dataset/{}'
        return self._api_get(definition.Dataset, path.format(datasetid))