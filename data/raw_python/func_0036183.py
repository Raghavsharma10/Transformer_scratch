def get_data(self, pivotrequest):
        """The method is getting data by pivot request"""

        path = '/api/1.0/data/pivot/'
        return self._api_post(definition.PivotResponse, path, pivotrequest)