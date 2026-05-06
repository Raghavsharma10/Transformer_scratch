def set_metadata(self, fp):
        """
        Set the XML metadata on a set.

        :param file fp: file-like object to read the XML metadata from.
        """
        base_url = self._client.get_url('SET', 'GET', 'single', {'id': self.id})
        self._manager._metadata.set(base_url, fp)

        # reload myself
        r = self._client.request('GET', base_url)
        return self._deserialize(r.json(), self._manager)