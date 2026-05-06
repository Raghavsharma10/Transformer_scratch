def update(self, id, data):
        """Update a component
        """
        id = self.as_id(id)
        response = self.http.patch(
            '%s/%s' % (self, id), json=data, auth=self.auth
        )
        response.raise_for_status()
        return response.json()