def set_metadata(self, set_id, fp):
        """
        Set the XML metadata on a set.

        :param file fp: file-like object to read the XML metadata from.
        """
        base_url = self.client.get_url('SET', 'GET', 'single', {'id': set_id})
        self._metadata.set(base_url, fp)