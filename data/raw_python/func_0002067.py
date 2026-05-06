def refresh(self):
        """
        Refresh this model from the server.

        Updates attributes with the server-defined values. This is useful where the Model
        instance came from a partial response (eg. a list query) and additional details
        are required.

        Existing attribute values will be overwritten.
        """
        r = self._client.request('GET', self.url)
        return self._deserialize(r.json(), self._manager)