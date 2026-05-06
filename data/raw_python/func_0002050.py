def save(self, with_data=False):
        """
        Edits this Source
        """
        r = self._client.request('PUT', self.url, json=self._serialize(with_data=with_data))
        return self._deserialize(r.json(), self._manager)