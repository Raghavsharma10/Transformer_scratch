def delete(self, resource_id):
        """ Deletes an existing resource

        Args:
            resource_id - int - The resource ID to be deleted
        """

        endpoint = '{}/{}'.format(self.endpoint, resource_id)

        response = self.api.execute("DELETE", endpoint)

        if not response.ok:
            raise Error.parse(response.json())

        return self._cls.parse(response.json())