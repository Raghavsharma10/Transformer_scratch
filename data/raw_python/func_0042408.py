def save(self, sync_only=False):
        """
        :param sync_only:
        :type: bool
        """

        entity = datastore.Entity(key=self._key)
        entity["last_accessed"] = self.last_accessed

        # todo: restore sync only
        entity["data"] = self._data
        if self.expires:
            entity["expires"] = self.expires

        self._client.put(entity)