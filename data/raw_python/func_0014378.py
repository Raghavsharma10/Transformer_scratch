def delete(self, resource_id, **kwargs):
        """
        Deletes a resource by ID.
        """

        return self.client._delete(self._url(resource_id), **kwargs)