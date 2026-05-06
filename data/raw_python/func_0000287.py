def delete(self, id):
        """Delete a component by id
        """
        id = self.as_id(id)
        response = self.http.delete(
            '%s/%s' % (self.api_url, id),
            auth=self.auth)
        response.raise_for_status()