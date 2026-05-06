def get_dimension(self, dataset, dimension):
        """The method is getting information about dimension with items"""

        path = '/api/1.0/meta/dataset/{}/dimension/{}'
        return self._api_get(definition.Dimension, path.format(dataset, dimension))