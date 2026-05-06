def get(self, id):
        """Get data for this component
        """
        id = self.as_id(id)
        url = '%s/%s' % (self, id)
        response = self.http.get(url, auth=self.auth)
        response.raise_for_status()
        return response.json()